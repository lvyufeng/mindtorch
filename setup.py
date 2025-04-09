import setuptools
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


import os
from setuptools.command.install import install

# 自定义安装类：在安装后生成 torch 的虚拟元数据
class PostInstallCommand(install):
    def run(self):
        # 1. 先执行标准安装流程
        install.run(self)

        # 2. 获取 Python 的 site-packages 路径
        import site
        site_packages = site.getsitepackages()[0]

        # 3. 创建 torch 的虚拟元数据目录
        torch_dist_info = os.path.join(site_packages, "torch-2.5.0.dist-info")
        os.makedirs(torch_dist_info, exist_ok=True)

        # 4. 写入 METADATA 文件（伪装成 PyTorch）
        metadata_content = """Metadata-Version: 2.1
Name: torch
Version: 2.5.0
Summary: Virtual torch provided by mindtorch
Home-page: https://github.com/your/mindtorch
Author: Your Name
Author-email: your@email.com
License: MIT
"""
        with open(os.path.join(torch_dist_info, "METADATA"), "w") as f:
            f.write(metadata_content)

        # 5. 可选：写入 RECORD 文件（避免 pip 报错）
        with open(os.path.join(torch_dist_info, "RECORD"), "w") as f:
            f.write("")


setuptools.setup(
    name="mindtorch",
    version="0.0.1",
    author="lvyufeng",
    author_email="lvyufeng@cqu.edu.cn",
    description="mindtorch project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lvyufeng/mindtorch",
    project_urls={
        "Bug Tracker": "https://github.com/lvyufeng/mindtorch/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    provides=["torch"],
    package_dir={"": "src"},
    packages=find_packages("src", exclude=("example", "tests")),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
    python_requires=">=3.9",
    install_requires=[
        "mindspore>=2.4",
        "requests",
    ],
    cmdclass={
        "install": PostInstallCommand,  # 绑定自定义安装逻辑
    },
)
