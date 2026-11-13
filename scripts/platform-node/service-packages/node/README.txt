GPU 节点安装说明

每台计算服务器安装一份。compute-node serve 负责本机加载与推理，compute-node connect 负责远程登记、上报和接单。适用环境为 Linux x86_64，glibc 要求取决于构建环境。

1. 解压与安装

先在下载目录用 sha256sum -c 校验对应的 .tar.gz.sha256，再解压到新目录。进入解包目录，用运行服务的普通 Linux 用户执行安装命令。

    bash ./install.sh

程序安装到 ~/.local/lib/compute-node/，两个 systemd 用户服务安装到 ~/.config/systemd/user/。

安装脚本需要 Python 3，并在 ~/.local/bin 注册短命令，同时设置 Bash 的 PATH。新终端可直接使用；当前终端执行 export PATH="$HOME/.local/bin:$PATH" 后即可使用。已有配置、模型和证书可继续使用。

2. 选择配置和数据目录

沿用现有节点身份、模型计划及制品时，继续使用原配置根和数据目录。将三项实际绝对路径写入 ~/.config/compute-node-service/service.env。

新节点可以沿用 unit 中的默认位置。先创建 ~/.config/compute-node-service，参照 node.example.json 填写节点 ID、调度地址、TLS 名称和证书绝对路径，保存为该目录的 node.json。节点私钥由运行账号保管，权限 0600。使用以下命令初始化独立身份，再准备对应的模型计划、GGUF 文件和兼容运行环境。

    XDG_CONFIG_HOME="$HOME/.config/compute-node-service" "$HOME/.local/lib/compute-node/compute-node" init --data-dir "$HOME/.local/state/compute-node"

使用 WSL 联调时保留到管理端的 SSH 转发。正式内网联调时在 node.json 中填写节点可以直接访问的调度地址。

3. 启动与查状态

迁移正在前台运行的节点时，先确认测试任务结束，依次退出原 connect 和 serve，再用以下命令启动后台服务并查看状态。

    systemctl --user daemon-reload
    systemctl --user enable --now compute-node-connect.service
    systemctl --user status compute-node.service compute-node-connect.service
    journalctl --user -u compute-node.service -u compute-node-connect.service -n 60 --no-pager

常用检查命令为 compute-node status、compute-node gpu list 和 compute-node model-list。短命令沿用节点 service.env 的配置目录和数据目录。

启动 connect 会依赖本机 serve。停止接单用 systemctl --user stop compute-node-connect.service。要一起停止，用 systemctl --user stop compute-node.service。退出登录仍需运行时，由管理员安排该账号的 lingering 或系统服务。

4. 升级与验收

节点停止接单并结束在跑任务后，停止节点服务。校验新包后备份原二进制和 unit，替换对应安装文件，保留节点配置、证书和数据目录，重新 daemon-reload 并启动。安装脚本可重复安装同一版本，发现不同文件会提示人工检查。

在管理机上用 compute-scheduler status 确认节点库存，再提交一条受支持的模型问答并核对回执。
