import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    pkg = get_package_share_directory('skid_robot_sim')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    urdf_path = os.path.join(pkg, 'urdf', 'robot.urdf.xacro')
    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    # corridor10_world.sdf を使用（同じ skid_robot_sim パッケージ内に配置）
    world_path = os.path.join(pkg, 'worlds', 'corridor10_world.sdf')
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items()
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # 廊下中央（X=0, Y=0）にスポーン
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'skid_robot',
            '-topic', '/robot_description',
            '-x', '1.0',
            '-y', '0.0',
            '-z', '0.1',
        ],
        output='screen'
    )

    bridge_config = os.path.join(pkg, 'config', 'bridge.yaml')
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': bridge_config,
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        gz_sim,
        rsp,
        spawn_robot,
        bridge,
    ])
