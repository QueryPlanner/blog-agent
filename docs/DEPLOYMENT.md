# Deployment Guide (Docker + GitHub Actions)

This guide details the automated deployment workflow: **Git Push -> GitHub Actions -> Your Server**.

## Prerequisites

1.  **Linux Server**: Ubuntu/Debian recommended.
2.  **Managed Postgres Database**: Connection string (e.g., Neon, AWS RDS, Supabase).
3.  **API Keys**: OpenRouter/Google, Langfuse (optional).
4.  **GitHub Repository**: You need admin access to configure secrets.

---

## Step 1: Prepare Your Server

We provide an automated script to provision your server with Docker, Firewall (UFW), and a dedicated user.

**Run on your server (as root):**

> [!WARNING]
> Review the script before executing.

```bash
curl -fsSL https://raw.githubusercontent.com/queryplanner/blog-agent/main/setup.sh | bash
```

This creates a user named `agent-runner` (or similar) and installs Docker.

---

## Step 2: Generate & Configure SSH Keys

To allow GitHub Actions to deploy to your server, you need a dedicated SSH key.

### 1. Generate a New Key Pair (Locally)
Run this on your local machine (not the server) to create a key without a passphrase:

```bash
ssh-keygen -t ed25519 -C "blog-agent-deploy" -f ~/.ssh/blog_agent_deploy -N ""
```

### 2. Install Public Key on Server
Copy the public key to your server's authorized keys for the user created in Step 1.

```bash
# Replace <user> with the user from setup.sh (default: agent-runner)
# Replace <host> with your server IP
ssh-copy-id -i ~/.ssh/blog_agent_deploy.pub <user>@<host>
```

*Verification:* Try logging in: `ssh -i ~/.ssh/blog_agent_deploy <user>@<host>`

---

## Step 3: Configure GitHub Secrets

You can sync your local environment variables to GitHub using the `gh` CLI.

### 1. Install GitHub CLI
If you haven't already: `brew install gh` (macOS) or see [installation docs](https://cli.github.com/).

### 2. Login
```bash
gh auth login
```

### 3. Set Secrets & Variables
Run the following commands to upload your configuration.

**Server Access Secrets:**
```bash
gh secret set SERVER_HOST --body "your.server.ip.address"
gh secret set SERVER_USER --body "agent-runner"
gh secret set SSH_PRIVATE_KEY < ~/.ssh/blog_agent_deploy
```

**Application Configuration (Secrets):**
*Ensure your `.env` file is populated with production values before running this, or set them manually.*

```bash
# Extract and set secrets from your local .env (or set manually)
gh secret set DATABASE_URL --body "$(grep DATABASE_URL .env | cut -d= -f2-)"
gh secret set OPENROUTER_API_KEY --body "$(grep OPENROUTER_API_KEY .env | cut -d= -f2-)"
gh secret set BLOG_GITHUB_TOKEN --body "$(grep BLOG_GITHUB_TOKEN .env | cut -d= -f2-)"
gh secret set ROOT_AGENT_MODEL --body "$(grep ROOT_AGENT_MODEL .env | cut -d= -f2-)"

# Optional: Observability
gh secret set LANGFUSE_PUBLIC_KEY --body "$(grep LANGFUSE_PUBLIC_KEY .env | cut -d= -f2-)"
gh secret set LANGFUSE_SECRET_KEY --body "$(grep LANGFUSE_SECRET_KEY .env | cut -d= -f2-)"
gh secret set LANGFUSE_HOST --body "$(grep LANGFUSE_HOST .env | cut -d= -f2-)"
```

**Application Configuration (Variables):**
```bash
gh variable set BLOG_REPO_OWNER --body "queryplanner"
gh variable set BLOG_REPO_NAME --body "blogs"
```

---

## Step 4: Deploy

1.  **Push to Main**: Any commit to the `main` branch triggers the deployment.
    ```bash
    git push origin main
    ```

2.  **Watch the Action**: Go to your GitHub repository -> **Actions** tab.

3.  **Verify**:
    *   The workflow will build the Docker image, push to GHCR, SSH into your server, and update the deployment.
    *   On your server, check status: `docker compose ps`

---

## Troubleshooting

### Permission Errors
If you see `PermissionError: [Errno 13] Permission denied: '/app/src/.adk'` in the logs:
1.  This typically happens if the named volume `agent_artifacts` was created by `root` in a previous run.
2.  **Fix**: SSH into the server and remove the volume to let Docker recreate it with correct permissions.
    ```bash
    docker compose down -v
    docker compose up -d
    ```
