"""历史导入路径兼容；新代码请使用 src.patch_attention。"""
try:
    from .patch_attention import CAReranker, CrossBlock, PatchCrossAttentionReranker
except ImportError:  # 兼容旧脚本将 src/ 直接放入 sys.path 的导入方式
    from patch_attention import CAReranker, CrossBlock, PatchCrossAttentionReranker
