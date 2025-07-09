#!/bin/bash

# 创建并激活 Python 3.10 虚拟环境
if [ ! -d "venv" ]; then
  python3.10 -m venv venv
fi
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 运行主脚本
python read_user_balance_head.py 
# ./venv/bin/pip install prophet