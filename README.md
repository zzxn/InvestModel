# 杂牌车队股票基本面模型

本模型用于『2020年复旦管院模拟证券投资大赛』的股票基本面分析。

* 队伍名称：价值投资杂牌车队
* 组合名称：日月光华精选组合

## 模型描述

参考 `股票基本面模型.pdf` 。

## 使用说明

### 环境

本模型由Python实现，要求：

* Python: 3.7+
* 其他包的要求在requirements.txt中，使用 `pip install -r requirements.txt`安装

### 配置

修改 `config.yml` 配置模型参数。

### 运行

将比赛系统（东方财富Choice）导出的特定格式的 `.xls` 文件放在/resources目录下，
然后运行main.py即可。

结果将出现在/out目录下。

`.xls` 文件的格式参考/resources下的示例。
