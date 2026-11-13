全局调度器安装说明

安装在 Linux 管理机。调度器运行不需要 GPU。适用环境为 Linux x86_64，glibc 要求取决于构建环境。

1. 解压与安装

先在下载目录用 sha256sum -c 校验对应的 .tar.gz.sha256，再解压到新目录。进入解包目录，用运行服务的普通 Linux 用户执行安装命令。

    bash ./install.sh

程序安装到 ~/.local/lib/compute-scheduler/，systemd 用户服务安装到 ~/.config/systemd/user/。安装完成后配置服务，再手动启动。

安装脚本需要 Python 3，并在 ~/.local/bin 注册短命令，同时设置 Bash 的 PATH。新终端可直接使用；当前终端执行 export PATH="$HOME/.local/bin:$PATH" 后即可使用。已有配置、模型和证书可继续使用。

2. 选择配置

已有部署可在 service.env 中设置 COMPUTE_SCHEDULER_CONFIG 指向原配置。

新部署时，把 scheduler.example.json 复制到 ~/.config/compute-scheduler/scheduler.json，将用户目录、证书路径、节点和模型计划换成实际值。提前创建配置中的 state_dir，属主为运行用户，权限 0700。使用既有测试 CA 签发证书，或用包内 remote-test-certs.sh 在新的目录生成短期联调证书。管理机保管 CA 私钥和调度器私钥。每个节点使用独立节点证书和私钥。

使用 SSH 转发时监听 127.0.0.1:8743。迁移到固定内网地址后，把监听地址和节点连接地址配置为实际可达的内网端点，并通过内网访问策略控制接入。

3. 启动与查状态

切换现有 WSL 调度器前，等 queued 和 dispatched 都归零，保存任务结果，再退出原前台调度器。然后用以下命令启动后台服务并查看状态。

    systemctl --user daemon-reload
    systemctl --user enable --now compute-scheduler.service
    systemctl --user status compute-scheduler.service
    journalctl --user -u compute-scheduler.service -n 40 --no-pager
    compute-scheduler status
    compute-scheduler task <任务UUID>

短命令从服务配置读取 state_dir。联调多套实例时可继续显式传入状态目录。停止服务用 systemctl --user stop compute-scheduler.service。

需要 HTTPS 网关时，用 gateway.example.json 配置独立的双向 TLS 证书和模型映射。网关与调度器使用同一个用户及状态目录。

    compute-scheduler gateway "$HOME/.config/compute-scheduler/gateway.json"

如需登出后仍运行或开机启动，请管理员按主机策略启用该服务账号的 lingering，或迁移到以专用普通账号运行的系统服务。WSL 发行版关闭时，内部服务也会停止。

4. 升级与验收

先清空当前队列并保存需要的结果，再停服务。校验新包后备份原二进制和 unit，替换对应安装文件，保留实际配置和证书，重新 daemon-reload 并启动。安装脚本可重复安装同一版本，发现不同文件会提示人工检查。

验收时查看节点和 GPU 库存，再通过 compute-scheduler submit 和 task 验证问答。调度器内存中的队列和短期结果路由会在重启时清空。HTTPS 网关受理的任务终态可从状态目录的 web-admissions 中查询。
