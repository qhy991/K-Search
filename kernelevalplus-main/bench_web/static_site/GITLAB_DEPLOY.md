# GitLab Pages 部署指南

本文档介绍如何将 KernelEvalPlus 静态网站部署到 GitLab Pages。

## 前置条件

1. GitLab 仓库（公开或私有均可）
2. 项目根目录已包含 `.gitlab-ci.yml` 配置文件
3. GitLab Runner 配置了 `docker` 和 `linux` tags

## CI/CD 架构

### 阶段 (Stages)

1. **build** - 构建静态网站和数据文件
2. **deploy** - 部署到 GitLab Pages 或预览环境

### 任务 (Jobs)

- `build:benchmark` - 生成静态数据并准备构建产物
- `pages` - 部署到 GitLab Pages（main 分支或 tag）
- `preview:benchmark` - 为 MR 创建预览环境
- `stop_preview` - 清理预览环境

## 部署步骤

### 1. 检查配置文件

确保项目根目录存在 `.gitlab-ci.yml` 文件，内容已配置好。

### 2. 提交并推送到 GitLab

```bash
# 添加配置文件
git add .gitlab-ci.yml bench_web/static_site/

# 提交
git commit -m "feat: Add GitLab Pages deployment"

# 推送到 GitLab（确保推送到 main 分支）
git push origin main
```

### 3. 查看部署状态

1. 进入你的 GitLab 项目页面
2. 点击左侧菜单 **CI/CD** → **Pipelines**
3. 查看最新的 pipeline 运行状态
   - **build:benchmark** - 构建阶段（约 1 分钟）
   - **pages** - 部署阶段（约 30 秒）
4. 等待两个阶段都完成

### 4. 访问网站

部署成功后，访问地址为：

```
https://<your-username>.gitlab.io/<your-project-name>/
```

例如：
- 如果用户名是 `qinhaiyan`
- 项目名是 `kernelevalplus`
- 访问地址：`https://qinhaiyan.gitlab.io/kernelevalplus/`

## 版本标签部署

支持通过 Git 标签触发部署：

```bash
# 创建版本标签
git tag -a v0.1.0 -m "Release v0.1.0: Initial benchmark site"

# 推送标签到 GitLab
git push origin v0.1.0
```

这将自动触发构建和部署流程。

### 5. 配置 GitLab Pages（首次部署）

如果是首次使用 GitLab Pages：

1. 进入项目设置：**Settings** → **Pages**
2. 查看 Pages 状态和访问地址
3. 确认 **Access Control** 设置（公开或私有访问）

## 本地预览

在推送之前，可以先本地预览：

```bash
# 生成数据文件
cd /home/qinhaiyan/kernelevalplus
python3 bench_web/static_site/generate_static.py

# 启动本地服务器
cd bench_web/static_site
python3 -m http.server 8000

# 在浏览器打开
# http://localhost:8000
```

## 自动更新

CI/CD pipeline 会在以下情况自动触发：

1. **推送到 main 分支** - 自动构建并部署到生产环境
2. **创建版本标签** - 自动部署标签版本
3. **创建 Merge Request** - 自动创建预览环境

每次触发时会自动：
1. 运行 `generate_static.py` 生成最新数据
2. 构建静态网站
3. 部署到 GitLab Pages

## 预览环境（Merge Request）

为 Merge Request 创建临时预览：

1. 创建新分支并推送
   ```bash
   git checkout -b feature/new-benchmark
   git push origin feature/new-benchmark
   ```

2. 在 GitLab 创建 Merge Request

3. CI/CD 自动创建预览环境

4. 在 MR 页面查看 **Environments** 标签页

5. MR 合并后，预览环境会在 1 周后自动清理

## GitLab Runner 要求

本项目的 CI/CD 配置需要 GitLab Runner 支持以下 tags：

- `docker` - 支持 Docker 容器运行
- `linux` - Linux 运行环境

如果你的 GitLab 实例没有配置这些 runners，请联系管理员或：

1. 移除 `.gitlab-ci.yml` 中的 `tags` 部分使用共享 runner
2. 或配置自己的 GitLab Runner

## 数据来源

静态网站使用以下数据：

1. **实验数据**: `/home/qinhaiyan/KERNELEVAL-exp/three_models_with_baseline_comparison.csv`
2. **Baseline 数据**: `core/tools/baseline_data_compact.json`

确保这些文件存在且是最新的。

## 故障排查

### Pipeline 失败

查看 pipeline 日志：
1. 进入 **CI/CD** → **Pipelines**
2. 点击失败的 pipeline
3. 查看 `pages` job 的日志

常见问题：
- 数据文件路径不正确
- Python 依赖缺失
- 权限问题

### Pages 没有更新

1. 确认 pipeline 成功完成
2. 等待几分钟（GitLab Pages 有缓存）
3. 清除浏览器缓存
4. 检查 GitLab Pages 设置是否启用

### 访问 404

1. 确认访问地址正确
2. 检查 `public/` 目录是否包含 `index.html`
3. 查看 GitLab Pages 设置中的实际地址

## 高级配置

### 自定义域名

在 **Settings** → **Pages** 中可以配置自定义域名：

1. 添加你的域名
2. 配置 DNS CNAME 记录指向 GitLab Pages
3. 等待 SSL 证书自动配置

### 访问控制

私有项目的 GitLab Pages 默认需要登录才能访问。

如果想公开访问：
- **Settings** → **General** → **Visibility**
- 设置为 **Public**

## 参考链接

- [GitLab Pages 官方文档](https://docs.gitlab.com/ee/user/project/pages/)
- [GitLab CI/CD 配置](https://docs.gitlab.com/ee/ci/yaml/)
