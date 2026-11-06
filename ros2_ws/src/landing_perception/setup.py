from setuptools import setup

package_name = "landing_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="GC-SHIRO",
    maintainer_email="GC-SHIRO@users.noreply.github.com",
    description="动态船载降落的 YOLO marker 检测节点",
    license="Proprietary",
    entry_points={"console_scripts": ["yolo_node = landing_perception.yolo_node:main"]},
)
