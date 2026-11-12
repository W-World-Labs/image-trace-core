import logging
import os
import uuid
import zipfile
from typing import List, Dict, Any, Tuple
from pathlib import Path
import io
import importlib.metadata
import re

logger = logging.getLogger(__name__)

# PDF is optional. Verify the official distribution before using its legacy
# import alias; the unrelated PyPI package named "fitz" is not this dependency.
try:
    importlib.metadata.version("PyMuPDF")
except importlib.metadata.PackageNotFoundError:
    fitz = None
else:
    try:
        import pymupdf as fitz
    except ModuleNotFoundError as exc:
        if exc.name != "pymupdf":
            raise
        import fitz  # Legacy import name from an installed official PyMuPDF.

# 图像处理
from PIL import Image as PILImage

from .image_processor import compute_image_features, is_image_file


class DocumentParser:
    """文档解析器，支持从PDF、DOCX、PPTX中提取图片"""

    def __init__(
        self, upload_dir: str = "data/uploads", extract_dir: str = "data/extracted"
    ):
        self.upload_dir = Path(upload_dir)
        self.extract_dir = Path(extract_dir)
        # data 目录（用于生成相对路径：uploads/...、extracted/...）
        self.base_dir = self.upload_dir.parent
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)

    def _rel_to_base(self, path: Path) -> str:
        """将路径转换为相对于 data 目录的路径字符串"""
        try:
            return str(path.resolve().relative_to(self.base_dir.resolve()))
        except Exception as exc:
            logger.debug("Path relativization failed for %s: %s", path, exc)
            return str(path)

    def extract_images_from_document(
        self, file_path: str, *, output_prefix: str | None = None
    ) -> List[Dict[str, Any]]:
        """
        从文档中提取所有图片

        Args:
            file_path: 文档文件路径

        Returns:
            提取的图片信息列表
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        output_prefix = output_prefix or uuid.uuid4().hex
        if not isinstance(output_prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", output_prefix):
            raise ValueError("output_prefix must contain 1-64 ASCII letters, digits, underscores or hyphens")
        if file_path.stat().st_size > 200 * 1024 * 1024:
            raise ValueError("document exceeds the 200 MiB input limit")

        if ext == ".pdf":
            return self._extract_from_pdf(file_path, output_prefix=output_prefix)
        elif ext in [".docx", ".pptx"]:
            return self._extract_from_office(file_path, output_prefix=output_prefix)
        else:
            raise ValueError(f"不支持的文档格式: {ext}")

    def _extract_from_pdf(
        self, pdf_path: Path, *, output_prefix: str | None = None
    ) -> List[Dict[str, Any]]:
        """从PDF文件提取图片"""
        extracted_images = []
        pdf_name = pdf_path.stem
        output_prefix = output_prefix or uuid.uuid4().hex

        doc = None
        try:
            if fitz is None:
                raise RuntimeError("PDF support is not installed; explicitly install image-trace-core[pdf]")
            # 方法1: 使用PyMuPDF提取嵌入图片
            doc = fitz.open(str(pdf_path))
            if len(doc) > 1000:
                raise ValueError("PDF exceeds the 1000-page limit")
            img_index = 0
            extracted_bytes = 0

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_info in image_list:
                    if img_index >= 1000:
                        raise ValueError("document exceeds the 1000-image limit")
                    xref = img_info[0]
                    smask = img_info[1] if len(img_info) > 1 else 0
                    if smask:
                        # Recombine the PDF image and its soft mask before saving.
                        # The resulting PNG preserves transparency; it intentionally
                        # does not include page content behind the transparent pixels.
                        base_pixmap = fitz.Pixmap(doc, xref)
                        mask_pixmap = fitz.Pixmap(doc, smask)
                        combined = fitz.Pixmap(base_pixmap, mask_pixmap)
                        image_bytes = combined.tobytes("png")
                        image_ext = "png"
                    else:
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            continue
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                    # Include a per-extraction token so repeated extraction of an
                    # identically named document cannot overwrite referenced files.
                    filename = f"{output_prefix}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
                    save_path = self.extract_dir / filename
                    extracted_bytes += len(image_bytes)
                    if extracted_bytes > 256 * 1024 * 1024:
                        raise ValueError("extracted images exceed the 256 MiB limit")

                    # 保存图片
                    with open(save_path, "xb") as f:
                        f.write(image_bytes)

                    # 如果是PDF格式，转换为JPEG
                    if image_ext.lower() == "pdf":
                        save_path = self._convert_to_jpeg(save_path)

                    # 计算特征
                    features = compute_image_features(str(save_path))
                    features.update(
                        {
                            "filename": save_path.name,
                            "file_path": self._rel_to_base(save_path),
                            "extracted_from": self._rel_to_base(pdf_path),
                            "extraction_method": "embedded",
                            "page_number": page_num + 1,
                            "image_index": img_index,
                        }
                    )

                    extracted_images.append(features)
                    img_index += 1

            # 方法2/3: 如果没有找到图片，或需要更多图片，将页面渲染为图片
            if not extracted_images:
                rendered_images = self._render_pdf_pages(pdf_path, pdf_name, output_prefix)
                if not rendered_images:
                    raise RuntimeError("PDF页面没有可渲染内容")
                extracted_images.extend(rendered_images)

        except Exception as e:
            # A failure after writing an artifact is a partial extraction. Fail
            # explicitly so process_document can remove only this call's files.
            if next(self.extract_dir.glob(f"{output_prefix}_*"), None):
                raise RuntimeError(f"PDF嵌入图片提取失败: {e}") from e
            # If embedded extraction is unavailable, render the complete pages.
            try:
                rendered_images = self._render_pdf_pages(pdf_path, pdf_name, output_prefix)
                if not rendered_images:
                    raise RuntimeError("PDF页面没有可渲染内容")
                extracted_images.extend(rendered_images)
            except Exception as e2:
                raise Exception(f"PDF解析失败: {str(e)}, 备用方法也失败: {str(e2)}")
        finally:
            if doc is not None:
                doc.close()

        return extracted_images

    def _render_pdf_pages(
        self, pdf_path: Path, pdf_name: str, output_prefix: str | None = None
    ) -> List[Dict[str, Any]]:
        """将PDF页面渲染为图片"""
        rendered_images = []
        output_prefix = output_prefix or uuid.uuid4().hex

        try:
            if fitz is None:
                raise RuntimeError("PDF support is not installed; explicitly install image-trace-core[pdf]")
            # 使用 PyMuPDF 直接渲染页面（不依赖 poppler/pdf2image）
            doc = fitz.open(str(pdf_path))
            try:
                if len(doc) > 1000:
                    raise ValueError("PDF exceeds the 1000-page limit")
                extracted_bytes = 0
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    if page.rect.width * page.rect.height * 4 > 40_000_000:
                        raise ValueError("rendered page exceeds the 40-million-pixel limit")

                    # 约等于 144dpi（默认 72dpi * 2）
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat, alpha=False)

                    filename = f"{output_prefix}_rendered_page{page_num + 1}.jpg"
                    save_path = self.extract_dir / filename

                    # 转成 JPEG 保存
                    img_bytes = pix.tobytes("png")
                    pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    with io.BytesIO() as encoded:
                        pil_img.save(encoded, "JPEG", quality=85, optimize=True)
                        data = encoded.getvalue()
                    extracted_bytes += len(data)
                    if extracted_bytes > 256 * 1024 * 1024:
                        raise ValueError("rendered images exceed the 256 MiB limit")
                    with save_path.open("xb") as output:
                        output.write(data)

                    # 计算特征
                    features = compute_image_features(str(save_path))
                    features.update(
                        {
                            "filename": filename,
                            "file_path": self._rel_to_base(save_path),
                            "extracted_from": self._rel_to_base(pdf_path),
                            "extraction_method": "rendered",
                            "page_number": page_num + 1,
                            "image_index": 0,
                        }
                    )

                    rendered_images.append(features)
            finally:
                doc.close()

        except Exception as e:
            if next(self.extract_dir.glob(f"{output_prefix}_*"), None):
                raise RuntimeError(f"渲染PDF页面失败: {str(e)}") from e
            logger.debug("渲染PDF页面失败: %s", e)

        return rendered_images

    def _extract_from_office(
        self, file_path: Path, *, output_prefix: str | None = None
    ) -> List[Dict[str, Any]]:
        """从Office文档（DOCX/PPTX）提取图片"""
        extracted_images = []
        file_name = file_path.stem
        file_type = file_path.suffix[1:].upper()  # 去掉点号，转为大写
        output_prefix = output_prefix or uuid.uuid4().hex

        try:
            # Office文档实际上是ZIP文件
            with zipfile.ZipFile(file_path, "r") as zip_file:
                # 查找媒体文件
                media_files = []
                for file_info in zip_file.filelist:
                    if file_info.filename.startswith(
                        "word/media/"
                    ) or file_info.filename.startswith("ppt/media/"):
                        media_files.append(file_info)

                if len(media_files) > 1000 or sum(info.file_size for info in media_files) > 256 * 1024 * 1024:
                    raise ValueError("Office media exceeds the 1000-file or 256 MiB limit")

                # 提取媒体文件
                for idx, file_info in enumerate(media_files):
                    # 读取文件数据
                    with zip_file.open(file_info) as source:
                        file_data = source.read(256 * 1024 * 1024 + 1)
                    if len(file_data) > 256 * 1024 * 1024:
                        raise ValueError("Office media exceeds the 256 MiB limit")

                    # 获取文件扩展名
                    original_name = os.path.basename(file_info.filename)
                    _, ext = os.path.splitext(original_name)
                    if not ext:
                        # 根据文件头判断类型
                        ext = self._guess_image_extension(file_data)

                    # Office media folders also contain video and audio. Accept
                    # only payloads Pillow can identify as a decodable image.
                    ext = ext.lower()
                    image_suffixes = {
                        ".jpg", ".jpeg", ".jpe", ".png", ".gif", ".bmp",
                        ".tif", ".tiff", ".webp", ".ico", ".jp2", ".avif",
                        ".heic", ".heif",
                    }
                    try:
                        with PILImage.open(io.BytesIO(file_data)) as source_image:
                            if source_image.width * source_image.height > 40_000_000:
                                raise ValueError("decoded image exceeds the 40-million-pixel limit")
                            source_image.verify()
                            detected_format = (source_image.format or "").lower()
                    except Exception as exc:
                        if ext in image_suffixes:
                            raise ValueError(
                                f"无法解码Office图片媒体: {file_info.filename}"
                            ) from exc
                        continue
                    format_ext = {
                        "jpeg": ".jpg", "png": ".png", "gif": ".gif",
                        "bmp": ".bmp", "tiff": ".tiff", "webp": ".webp",
                        "ico": ".ico", "jp2": ".jp2", "avif": ".avif",
                        "heic": ".heic", "heif": ".heif",
                    }.get(detected_format)
                    if not format_ext:
                        continue
                    ext = format_ext
                    filename = f"{output_prefix}_media{idx + 1}{ext}"
                    save_path = self.extract_dir / filename

                    # 保存文件
                    with open(save_path, "xb") as f:
                        f.write(file_data)

                    # 计算特征
                    features = compute_image_features(str(save_path))
                    features.update(
                        {
                            "filename": save_path.name,
                            "file_path": self._rel_to_base(save_path),
                            "extracted_from": self._rel_to_base(file_path),
                            "extraction_method": f"{file_type}_media",
                            "page_number": None,
                            "image_index": idx,
                            "media_path": file_info.filename,
                        }
                    )

                    extracted_images.append(features)

        except Exception as e:
            raise Exception(f"从{file_type}文档提取图片失败: {str(e)}")

        return extracted_images

    def _guess_image_extension(self, file_data: bytes) -> str:
        """根据文件头猜测图像格式"""
        if file_data.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        elif file_data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        elif file_data.startswith(b"GIF87a") or file_data.startswith(b"GIF89a"):
            return ".gif"
        elif file_data.startswith(b"BM"):
            return ".bmp"
        elif file_data.startswith(b"II*\x00") or file_data.startswith(b"MM\x00*"):
            return ".tiff"
        elif file_data.startswith(b"%PDF"):
            return ".pdf"
        else:
            return ""

    def _convert_to_jpeg(self, image_path: Path) -> Path:
        """将图片转换为JPEG格式"""
        try:
            with PILImage.open(image_path) as img:
                # 转换为RGB模式（如果需要）
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # 生成新文件名
                new_path = image_path.with_suffix(".jpg")

                # 保存为JPEG
                img.save(new_path, "JPEG", quality=85, optimize=True)

                # 删除原文件
                if new_path != image_path:
                    image_path.unlink()

                return new_path
        except Exception as e:
            print(f"转换图片为JPEG失败: {str(e)}")
            return image_path

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        处理文档文件，返回提取结果

        Args:
            file_path: 文档文件路径

        Returns:
            处理结果字典
        """
        file_path = Path(file_path)

        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 检查文件类型
        ext = file_path.suffix.lower()
        if ext not in [".pdf", ".docx", ".pptx"]:
            raise ValueError(f"不支持的文档格式: {ext}")

        # 提取图片
        try:
            output_prefix = uuid.uuid4().hex
            extracted_images = self.extract_images_from_document(
                file_path, output_prefix=output_prefix
            )

            return {
                "status": "success",
                "document_path": self._rel_to_base(file_path),
                "document_name": file_path.name,
                "document_type": ext[1:].upper(),
                "extracted_count": len(extracted_images),
                "images": extracted_images,
            }

        except Exception as e:
            # Extraction is an atomic operation from the caller's point of view.
            # Only this call's uniquely prefixed artifacts are removed.
            if "output_prefix" in locals():
                for artifact in self.extract_dir.glob(f"{output_prefix}_*"):
                    try:
                        artifact.unlink(missing_ok=True)
                    except OSError:
                        logger.exception("Failed to clean partial extraction %s", artifact)
            return {
                "status": "error",
                "document_path": self._rel_to_base(file_path),
                "error": str(e),
                "extracted_count": 0,
                "images": [],
            }
