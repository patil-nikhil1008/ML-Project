#!/usr/bin/env python3
"""Automated script to create a GitHub repository (if not exists) and push code.

Usage:
    python scripts/create_and_push_github.py <github_username> <personal_access_token> [--private]
"""

import sys
import subprocess
import requests


def create_and_push(username: str, token: str, repo_name: str = "photobomb-detector", is_private: bool = False):
    print(f"[*] Checking GitHub account for user '{username}' and repository '{repo_name}'...")

    # 1. Create repository via GitHub REST API
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PhotobombDetector-Deployer",
    }
    payload = {
        "name": repo_name,
        "private": is_private,
        "description": "Photobomb Detection & Subject Classification Pipeline — YOLOv8, 5-Signal Fusion, Web Studio & AI Inpainting",
        "has_issues": True,
        "has_projects": True,
        "has_wiki": False,
    }

    resp = requests.post("https://api.github.com/user/repos", json=payload, headers=headers)

    if resp.status_code == 201:
        print(f"[+] Successfully created repository: https://github.com/{username}/{repo_name}")
    elif resp.status_code == 422:
        print(f"[*] Repository '{repo_name}' already exists on GitHub. Proceeding to push...")
    else:
        print(f"[!] GitHub API Error ({resp.status_code}): {resp.text}")
        sys.exit(1)

    # 2. Configure Git remote
    # Strip token from printed logs for security
    authenticated_remote = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"
    public_remote = f"https://github.com/{username}/{repo_name}.git"

    # Check if remote exists
    remotes = subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.split()
    if "origin" in remotes:
        subprocess.run(["git", "remote", "set-url", "origin", authenticated_remote], check=True)
    else:
        subprocess.run(["git", "remote", "add", "origin", authenticated_remote], check=True)

    print("[*] Remote origin configured. Ensuring branch is 'main'...")
    subprocess.run(["git", "branch", "-M", "main"], check=True)

    print("[*] Pushing code, test suite, reports, and benchmarks to GitHub...")
    push_res = subprocess.run(["git", "push", "-u", "origin", "main"], capture_output=True, text=True)

    # Sanitize remote URL back to public URL (remove token from .git/config for security)
    subprocess.run(["git", "remote", "set-url", "origin", public_remote], check=True)

    if push_res.returncode == 0:
        print(f"\n✨ SUCCESS! Project is live on GitHub: {public_remote}")
    else:
        print(f"\n[!] Push failed:\n{push_res.stderr}")
        sys.exit(push_res.returncode)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/create_and_push_github.py <username> <token> [--private]")
        sys.exit(1)

    user = sys.argv[1]
    pat = sys.argv[2]
    priv = "--private" in sys.argv

    create_and_push(user, pat, is_private=priv)
