from setuptools import setup

package_name = 'door_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'door_detector_node = door_detector.door_detector_node:main',
            'odom_to_tf = door_detector.odom_to_tf:main',
        ],
    },
)
