# Anno Server

图像标注服务端，提供 REST API，支持 2D 标注、标签管理、AI 自动标注与交互式标注。

容器镜像：`ghcr.io/anno-universe/anno-server:latest`

## 快速开始

### 前置要求

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/)

### 1. 准备配置文件

```bash
cp docker-compose.example.yml docker-compose.yml
cp Caddyfile.example Caddyfile
cp .env.example .env
```

### 2. 编辑 `.env`

```ini
# 必填：生成一个随机密钥（不要使用默认值）
SECRET_KEY=your-random-secret-key

# 必填：替换为你的域名
ALLOWED_HOSTS=your-domain.com

# 可选：时区
TZ=Asia/Shanghai
```

### 3. 启动

```bash
docker compose up -d
```

### 4. 验证

```bash
curl http://localhost/health/
# 返回: OK
```

首次启动会自动执行数据库迁移和静态文件收集，稍等片刻即可。

## 配置说明

### 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `SECRET_KEY` | **是** | — | Django 密钥，同时用作 JWT 签名密钥 |
| `ALLOWED_HOSTS` | **是** | `*` | 允许访问的域名，多个用逗号分隔 |
| `DEBUG` | 否 | `false` | 生产环境务必设为 `false` |
| `TZ` | 否 | `Asia/Shanghai` | 时区 |
| `INTERACTIVE_SESSION_TOKEN_TTL_SECONDS` | 否 | `1800` | 交互式标注会话令牌有效期（秒） |

### Caddyfile（反向代理）

默认 Caddyfile 监听 `:80`，处理：

- `/health/` — 健康检查
- `/static/*` — 静态文件（由 Caddy 直接返回）
- 其他请求 — 反向代理到 Django

如需 **HTTPS**，将 `:80` 改为你的域名，Caddy 会自动申请 Let's Encrypt 证书：

```caddyfile
your-domain.com {
    # ... 其余配置不变
}
```

## 服务架构

部署包含 4 个容器：

| 服务 | 镜像 | 说明 |
|------|------|------|
| **django** | `ghcr.io/anno-universe/anno-server:latest` | Gunicorn Web 服务（4 worker，端口 8000，内部） |
| **db_worker** | `ghcr.io/anno-universe/anno-server:latest` | 后台任务轮询（自动标注等异步任务） |
| **postgres** | `postgres:16-alpine` | PostgreSQL 数据库（端口 5432，内部） |
| **caddy** | `caddy:2-alpine` | 反向代理（端口 80/443），处理 HTTPS |

### 数据持久化

| 卷 | 内容 | 备份建议 |
|----|------|----------|
| `postgres_data` | 数据库数据 | **必须备份**（pg_dump 或卷快照） |
| `media_data` | 用户上传的图片（核心数据） | **必须备份** |
| `/tmp/anno-thumbnails` | 缩略图缓存（挂载宿主机 `/tmp`） | 无需备份 |
| `caddy_data` | TLS 证书 | 建议备份（避免频繁申请证书） |

## 运维

### 创建管理员账号

```bash
docker compose exec django python manage.py createsuperuser
```

之后可通过 Django Admin（`/admin/`）管理项目和用户。

### 升级镜像

```bash
docker compose pull
docker compose up -d
```

新版本启动时会自动执行数据库迁移。

### 导出 API 文档

```bash
docker compose exec django python manage.py export_openapi
```

### 备份数据库

```bash
docker compose exec postgres pg_dump -U postgres anno > backup.sql
```

### 备份媒体文件

```bash
docker compose cp django:/app/media ./media-backup
```

### 查看日志

```bash
# 所有服务
docker compose logs -f

# 指定服务
docker compose logs -f django
docker compose logs -f db_worker
```

## API 概览

API 统一以 `/api/` 为前缀，认证方式为 JWT。

主要端点分类：

| 模块 | 路径 | 说明 |
|------|------|------|
| 认证 | `/api/token/pair` | 获取 JWT 令牌 |
| 用户 | `/api/users/` | 注册、个人信息 |
| 项目 | `/api/projects/` | 项目管理、成员、API Key |
| 图片 | `/api/projects/{id}/images/` | 图片上传、缩略图 |
| 标注 | `/api/projects/{id}/images/{id}/annotations/` | 多边形/矩形/关键点标注（不可变模式） |
| 标签 | `/api/projects/{id}/tags/` | 项目级标签定义与统计 |
| 自动标注 | `/api/projects/{id}/auto-annotate/` | 批量/单图 AI 标注、运行状态、重试 |
| 交互式标注 | `/api/projects/{id}/images/{id}/interactive-sessions/` | SAM 风格交互会话 |

标注操作均保留完整审计记录（谁、何时、何种操作、来自人工还是 AI）。
