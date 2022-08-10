"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "529a3d87a799a2cba29bc1d0f426a00d5bb4c88f"
    LLVM_SHA256 = "3d2ca52bd36ad3904f2f5d0e43935b0e82f3d1ac137e0a89025141e36735944f"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm:llvm.BUILD",
        patch_file = [
            "//third_party/llvm:infer_type.patch",  # TODO(b/231285230): remove once resolved
            "//third_party/llvm:build.patch",
            "//third_party/llvm:macos_build_fix.patch",
            "//third_party/llvm:0001-mlir-ROCm-Add-shfl.sync.bfly-lowering.patch",
        ],
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
