"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "8c712296fb75ff73db08f92444b35c438c01a405"
    LLVM_SHA256 = "fc4c884b886a001275c7753dedebc005e0d16eb53115d9f63dc8fccc348e3074"

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
            "//third_party/llvm:generated.patch",  # Autogenerated, don't remove.
            "//third_party/llvm:build.patch",
            "//third_party/llvm:mathextras.patch",
            "//third_party/llvm:toolchains.patch",
            "//third_party/llvm:0001-mlir-ROCm-Add-shfl.sync.bfly-lowering.patch",
            "//third_party/llvm:0001-llvm-nvptx-Fix-error-GVN-on-shared-memory-load.patch",
            "//third_party/llvm:0001-mlir-not-fold-UnrealizedConversionCastOp-with-ui-si.patch",
        ],
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
