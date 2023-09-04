import os
import yaml

def main():

    # 获取当前环境的包列表
    packages = os.popen('pip list --format=freeze').read().split("\n")[:-1]

    # 将包列表写入YAML文件
    with open('VISoRMap.yml', 'w') as outfile:
        yaml.dump({'dependencies': packages}, outfile, default_flow_style=False)

if __name__ == "__main__":
    main()