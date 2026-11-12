"""Local image decoding, perceptual hashes and stateless descriptor helpers.

W-owned Image Trace code; MIT (see LICENSE). Third-party libraries are imported
from separately installed original packages, never vendored or patched here.
"""
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imagehash
import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass  # pillow-heif 未安装时静默跳过

try:
    import rawpy as _rawpy
except ImportError:
    _rawpy = None

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf"}

def load_image_rgb(image_path: str) -> PILImage.Image:
    """Decode any Pillow-supported image to an RGB image for vision algorithms.

    Scientific integer and floating point grayscale images are linearly mapped
    from their finite min/max to 8-bit. Non-finite samples map to the low/high
    endpoint (NaN to low); all-non-finite and constant images become black.
    Ordinary 8-bit images retain Pillow's existing RGB conversion semantics.
    The source file is only read and is never modified.
    """
    _, ext = os.path.splitext(str(image_path).lower())
    try:
        if ext in RAW_EXTENSIONS and _rawpy is not None:
            with _rawpy.imread(str(image_path)) as raw:
                return PILImage.fromarray(raw.postprocess()).convert("RGB")

        with PILImage.open(image_path) as source:
            source.seek(0)  # deterministic first frame for animated formats
            if source.width * source.height > 40_000_000:
                raise ValueError("decoded image exceeds the 40-million-pixel limit")
            if source.mode in {"I;16", "I;16L", "I;16B", "I", "F"}:
                values = np.asarray(source)
                finite = np.isfinite(values)
                if not finite.any():
                    gray = np.zeros(values.shape, dtype=np.uint8)
                else:
                    low = float(values[finite].min())
                    high = float(values[finite].max())
                    safe = np.nan_to_num(values.astype(np.float64), nan=low,
                                         posinf=high, neginf=low)
                    if high <= low:
                        gray = np.zeros(values.shape, dtype=np.uint8)
                    else:
                        scaled = np.clip((safe - low) * (255.0 / (high - low)), 0, 255)
                        gray = np.rint(scaled).astype(np.uint8)
                return PILImage.fromarray(gray).convert("RGB")
            return source.convert("RGB")
    except Exception as exc:
        raise ValueError(f"无法读取图像文件 {image_path}: {exc}") from exc

def _read_cv2_rgb(image_path: str) -> np.ndarray:
    """Return normalized RGB pixels from the shared decoder."""
    return np.asarray(load_image_rgb(image_path))

def _read_cv2_gray(image_path: str) -> np.ndarray:
    _ensure_cv2()
    return cv2.cvtColor(_read_cv2_rgb(image_path), cv2.COLOR_RGB2GRAY)

def _read_cv2_bgr(image_path: str) -> np.ndarray:
    _ensure_cv2()
    return cv2.cvtColor(_read_cv2_rgb(image_path), cv2.COLOR_RGB2BGR)

try:
    import cv2
except Exception:
    cv2 = None  # 延迟到使用特征匹配类算法时再报错

SUPPORTED_IMAGE_EXTENSIONS = {
    # 常用格式
    ".jpg",
    ".jpeg",
    ".jfif",
    ".jpe",
    ".png",
    ".apng",
    ".gif",
    ".bmp",
    ".dib",
    ".tif",
    ".tiff",
    ".webp",
    # JPEG 2000 系列
    ".jp2",
    ".j2k",
    ".j2c",
    ".jpf",
    ".jpx",
    ".jpc",
    # 专业/设计格式
    ".psd",  # Adobe Photoshop
    ".eps",
    ".ps",  # Encapsulated PostScript
    # 图标格式
    ".ico",
    ".cur",
    ".icns",
    # Targa 系列
    ".tga",
    ".vda",
    ".icb",
    ".vst",
    # 科学图像
    ".fits",
    ".fit",
    ".fts",
    # 传统格式
    ".pcx",
    ".dcx",
    ".dds",  # DirectDraw Surface
    ".qoi",  # Quite Ok Image
    ".mpo",  # Multi-Picture Object
    # NetPBM 系列
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
    # SGI 系列
    ".sgi",
    ".rgb",
    ".rgba",
    ".bw",
    # Sun Raster
    ".ras",
    # 动画帧
    ".flc",
    ".fli",
    # X Window
    ".xbm",
    ".xpm",
    # RAW 相机格式（需要 rawpy/libraw，记录支持但降级处理）
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".dng",
    ".orf",
    ".rw2",
    ".raf",
    # HEIF/AVIF（需要 pillow-heif/pillow-avif 插件）
    ".heic",
    ".heif",
    ".avif",
}

def compute_file_md5(file_path: str) -> str:
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def compute_image_features(image_path: str) -> Dict[str, Any]:
    """
    计算图像的多种哈希值和基本信息

    Args:
        image_path: 图像文件路径

    Returns:
        包含各种哈希值和图像信息的字典
    """
    features = {}

    # 文件哈希
    features["file_hash"] = compute_file_md5(image_path)
    features["file_size"] = os.path.getsize(image_path)

    _, ext = os.path.splitext(image_path.lower())

    try:
        img = load_image_rgb(image_path)
        with img:
            # 获取图像尺寸
            features["width"], features["height"] = img.size

            # 转换为RGB模式（如果需要）
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 计算各种感知哈希
            features["phash"] = str(imagehash.phash(img, hash_size=8))
            features["dhash"] = str(imagehash.dhash(img, hash_size=8))
            features["ahash"] = str(imagehash.average_hash(img, hash_size=8))
            features["whash"] = str(imagehash.whash(img, hash_size=8))
            features["colorhash"] = str(imagehash.colorhash(img))

    except Exception as e:
        raise ValueError(f"无法处理图像文件 {image_path}: {str(e)}")

    return features

def hamming_distance(hash1: str, hash2: str) -> int:
    """
    计算两个哈希值的汉明距离

    Args:
        hash1, hash2: 十六进制哈希字符串

    Returns:
        汉明距离（不同位的数量）
    """
    if len(hash1) != len(hash2):
        raise ValueError("哈希长度不一致，无法计算汉明距离")
    # 将十六进制转换为二进制
    bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
    bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

    # 计算不同位的数量
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))

def calculate_similarity(hash1: str, hash2: str, max_bits: Optional[int] = None) -> float:
    """
    计算两个哈希值的相似度

    Args:
        hash1, hash2: 十六进制哈希字符串
        max_bits: 哈希有效位数；默认按哈希文本推断（colorhash 为42位）

    Returns:
        相似度分数 (0-1，1表示完全相同)
    """
    distance = hamming_distance(hash1, hash2)
    if max_bits is None:
        # imagehash.ColorHash uses 14x3 = 42 significant bits, rendered as
        # eleven hex digits (the final two padding bits are always zero).
        max_bits = 42 if len(hash1) == 11 else len(hash1) * 4
    if max_bits <= 0 or distance > max_bits:
        raise ValueError("无效的哈希位宽")
    similarity = (max_bits - distance) / max_bits
    return max(0.0, min(1.0, similarity))

def _ensure_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not installed; explicitly install image-trace-core[vision]."
        )

def _create_extractor(algo: str, max_features: int = 512):
    """
    创建特征提取器，返回 (extractor, norm_type)
    支持: orb / brisk / sift / akaze / kaze
    """
    _ensure_cv2()
    if isinstance(max_features, bool) or not isinstance(max_features, int) or not 1 <= max_features <= 10000:
        raise ValueError("max_features must be an integer between 1 and 10000")
    algo = algo.lower()
    if algo == "orb":
        return cv2.ORB_create(
            nfeatures=max_features, scaleFactor=1.2, nlevels=8
        ), cv2.NORM_HAMMING
    elif algo == "brisk":
        return cv2.BRISK_create(), cv2.NORM_HAMMING
    elif algo == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("当前 OpenCV 未包含 SIFT。")
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    elif algo == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    elif algo == "kaze":
        return cv2.KAZE_create(), cv2.NORM_L2
    else:
        raise ValueError(f"不支持的特征算法: {algo}")

def compute_descriptor(image_path: str, algo: str, max_features: int = 512):
    """
    计算局部特征描述子，支持 orb/sift/brisk/akaze/kaze
    返回 (descriptor, norm_type)
    """
    _ensure_cv2()
    img = _read_cv2_gray(image_path)

    extractor, norm = _create_extractor(algo, max_features)
    _, desc = extractor.detectAndCompute(img, None)
    return desc, norm

def compute_descriptor_with_kp(image_path: str, algo: str, max_features: int = 512):
    """
    返回 (keypoints, descriptor, norm)
    """
    _ensure_cv2()
    img = _read_cv2_gray(image_path)

    extractor, norm = _create_extractor(algo, max_features)
    kps, desc = extractor.detectAndCompute(img, None)
    return kps, desc, norm

def calculate_descriptor_similarity(desc1, desc2, norm: int, top_k: int = 64) -> float:
    """
    通用局部特征相似度（BFMatcher）
    返回 0-1，越大越相似。
    """
    _ensure_cv2()
    if desc1 is None or desc2 is None:
        return 0.0
    if norm == cv2.NORM_HAMMING and desc1.shape[1] != desc2.shape[1]:
        raise ValueError("二进制描述子位宽不一致")
    matcher = cv2.BFMatcher(norm, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    if not matches:
        return 0.0
    matches = sorted(matches, key=lambda m: m.distance)[:top_k]
    avg_dist = sum(m.distance for m in matches) / len(matches)
    # Normalize binary descriptor distances by their real number of bits.
    if norm == cv2.NORM_HAMMING:
        bit_width = int(desc1.shape[1]) * 8
        similarity = 1.0 - min(max(avg_dist, 0.0), float(bit_width)) / bit_width
    else:
        similarity = 1.0 - min(max(avg_dist, 0.0), 512.0) / 512.0
    return max(0.0, min(1.0, similarity))

def find_similar_images(
    query_image: Dict[str, Any],
    image_list: List[Dict[str, Any]],
    threshold: float = 0.85,
    hash_type: str = "phash",
) -> List[Tuple[Dict[str, Any], float]]:
    """
    查找与查询图像相似的图像

    Args:
        query_image: 查询图像的特征字典
        image_list: 待比对的图像列表
        threshold: 相似度阈值（0-1）
        hash_type: 使用的哈希类型（'phash', 'dhash', 'ahash', 'whash'）

    Returns:
        包含相似图像和相似度分数的列表，按相似度降序排列
    """
    query_hash = query_image.get(hash_type)
    if not query_hash:
        raise ValueError(f"查询图像缺少 {hash_type} 哈希值")

    similar_images = []

    for img in image_list:
        img_hash = img.get(hash_type)
        if not img_hash:
            continue

        # 跳过与自身比较
        if img is query_image or (
            query_image.get("id") is not None and img.get("id") == query_image["id"]
        ):
            continue

        similarity = calculate_similarity(query_hash, img_hash)

        if similarity >= threshold:
            similar_images.append((img, similarity))

    # 按相似度降序排列
    similar_images.sort(key=lambda x: x[1], reverse=True)

    return similar_images

def group_similar_images(
    images: List[Dict[str, Any]], threshold: float = 0.85, hash_type: str = "phash"
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    将相似的图像分组

    Args:
        images: 图像列表
        threshold: 相似度阈值
        hash_type: 使用的哈希类型

    Returns:
        (相似组列表, 未分组图像列表)
    """
    if not images:
        return [], []

    # 复制列表，避免修改原数据
    remaining_images = images.copy()
    groups = []
    processed_indices = set()

    for i, query_img in enumerate(images):
        # 跳过已处理的图像
        if i in processed_indices:
            continue

        # 查找相似图像
        similar = find_similar_images(query_img, remaining_images, threshold, hash_type)

        if similar:
            # 创建新组，包含查询图像和所有相似图像
            group = [query_img] + [img for img, _ in similar]
            groups.append(group)

            # 标记已处理的图像
            processed_indices.add(i)
            for img, _ in similar:
                # Dict instances in this list are the actual records; duplicate
                # content hashes still represent distinct images.
                processed_indices.add(next(j for j, candidate in enumerate(images) if candidate is img))

    # 找出未分组的图像
    ungrouped = [img for i, img in enumerate(images) if i not in processed_indices]

    return groups, ungrouped

def is_image_file(file_path: str) -> bool:
    """检查文件是否为支持的图像格式（全品种）"""
    _, ext = os.path.splitext(file_path.lower())
    return ext in SUPPORTED_IMAGE_EXTENSIONS

def resize_image_if_needed(
    image_path: str, max_size: int = 1024, output_path: str = None
) -> str:
    """
    如果图像尺寸过大，调整图像大小

    Args:
        image_path: 输入图像路径
        max_size: 最大尺寸（宽度或高度）
        output_path: 输出路径（如果为None，则覆盖原文件）

    Returns:
        处理后的图像路径
    """
    with PILImage.open(image_path) as img:
        width, height = img.size

        # 如果图像尺寸在限制内，直接返回
        if max(width, height) <= max_size:
            return image_path

        # 计算新尺寸
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        # 调整大小
        resized_img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

        # 保存图像
        output = output_path or image_path
        resized_img.save(output, quality=85, optimize=True)

        return output
