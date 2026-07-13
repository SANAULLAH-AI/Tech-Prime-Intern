---
title: Git Cheat Sheet
subtitle: Ultimate command reference for developers
date: 2026-07-13 10:43:24
background: bg-[#d7593e]
icon: git
tags:
  - github
  - gitlab
  - version-control
  - vcs
  - devops
categories:
  - Developer Tools
intro: |
  An advanced, production-ready Git command-line reference guide.
  Covers everything from basic staging to advanced branching, stashing, and cloud deployment.
plugins:
  - copyCode
  - search
  - darkMode
---

## Repository Management

### Initialize & Clone

Create a new local repository
```shell script
$ git init [project name]
```

Clone a repository
```shell script
$ git clone git_url
```

Clone a repository into a specified directory
```shell script
$ git clone git_url my_directory
```

### Configuration

Set the name that will be attached to your commits and tags
```shell script
$ git config --global user.name "name"
```

Set an email address that will be attached to your commits and tags
```shell script
$ git config --global user.email "email"
```

Enable some colorization of Git output
```shell script
$ git config --global color.ui auto
```

Edit the global configuration file in a text editor
```shell script
$ git config --global --edit
```

---

## Making Changes

### Stage & Commit

Show modified files in working directory, staged for your next commit
```shell script
$ git status
```

Stages the file, ready for commit
```shell script
$ git add [file]
```

Stage all changed files, ready for commit
```shell script
$ git add .
```

Commit all staged files to version history
```shell script
$ git commit -m "commit message"
```

Commit all your tracked files to version history
```shell script
$ git commit -am "commit message"
```

### Undo & Reset

Discard changes in working directory which is not staged
```shell script
$ git restore [file]
```

Unstage a staged file or file which is staged
```shell script
$ git restore --staged [file]
```

Unstage a file, keeping the file changes
```shell script
$ git reset [file]
```

Revert everything to the last commit
```shell script
$ git reset --hard
```

### Compare Changes

Diff of what is changed but not staged
```shell script
$ git diff
```

Diff of what is staged but not yet committed
```shell script
$ git diff --staged
```

Apply any commits of current branch ahead of specified one
```shell script
$ git rebase [branch]
```

### Track Files

Delete the file from project and stage the removal for commit
```shell script
$ git rm [file]
```

Change an existing file path and stage the move
```shell script
$ git mv [existing-path] [new-path]
```

### Ignore Files

A `.gitignore` file specifies intentionally untracked files that Git should ignore
```
/logs/*
!logs/.gitkeep
.DS_store
node_modules
.sass-cache
```

---

## Branch Management

### Create & Switch

List all local branches
```shell script
$ git branch
```

List all branches, local and remote
```shell script
$ git branch -av
```

List all branches and their upstreams
```shell script
$ git branch -vv
```

Get only remote branches
```shell script
$ git branch -r
```

Switch to my_branch, and update working directory
```shell script
$ git checkout my_branch
```

Quickly switch to the previous branch
```shell script
$ git checkout -
```

Create a new branch called new_branch
```shell script
$ git checkout -b new_branch
```

### Merge & Delete

Merge branchA into branchB
```shell script
$ git checkout branchB
$ git merge branchA
```

Delete the branch called my_branch
```shell script
$ git branch -d my_branch
```

### Working with Files

Checkout a single file from another branch
```shell script
$ git checkout <branch> -- <file>
```

Tag the current commit
```shell script
$ git tag my_tag
```

### Rename Branch

```shell script
# Rename to new_name
$ git branch -m <new_name>

# Push and reset
$ git push origin -u <new_name>

# Delete remote branch
$ git push origin --delete <old>
```

---

## Viewing History

### Basic Log

Show the commit history for the currently active branch
```shell script
$ git log
```

Show the commits on branchA that are not on branchB
```shell script
$ git log branchB..branchA
```

Show the commits that changed file, even across renames
```shell script
$ git log --follow [file]
```

### Visual Log

Print out a cool visualization of your log
```shell script
$ git log --pretty=oneline --graph --decorate --all
```

### Search & Filter

Search change by content
```shell script
$ git log -S'<a term in the source>'
```

Show changes over time for specific file
```shell script
$ git log -p <file_name>
```

### Show Details

Show the diff of what is in branchA that is not in branchB
```shell script
$ git diff branchB...branchA
```

Show any object in Git in human-readable format
```shell script
$ git show [SHA]
```

---

## Stashing Changes

Save modified and staged changes
```shell script
$ git stash
```

List stack-order of stashed file changes
```shell script
$ git stash list
```

Write working from top of stash stack
```shell script
$ git stash pop
```

Discard the changes from top of stash stack
```shell script
$ git stash drop
```

---

## Git Tricks

### Rewrite History

Rewrite last commit message
```shell script
$ git commit --amend -m "new message"
```

Amend the latest commit without changing the commit message
```shell script
$ git commit --amend --no-edit
```

### Git Aliases

```shell script
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| Initialize repo | `git init` |
| Clone repo | `git clone <url>` |
| Check status | `git status` |
| Stage file | `git add <file>` |
| Commit | `git commit -m "msg"` |
| Create branch | `git branch <name>` |
| Switch branch | `git checkout <name>` |
| Merge branch | `git merge <branch>` |
| View log | `git log` |
| View diff | `git diff` |
| Stash changes | `git stash` |
| Tag commit | `git tag <name>` |

---

> 💡 **Pro Tip**: Use `git --help` or `git <command> --help` to learn more about any Git command.

---

## Author & Maintainer

Crafted with ❤️ by **Sanaullah** — dedicated to AI, open-source development, and building robust version control workflows.

### Connect With Me

[![GitHub](https://img.shields.io/badge/GitHub-sanaullah--ai-181717?style=flat&logo=github)](https://github.com/sanaullah-ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-sanaullah7964-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/sanaullah7964)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sana%20Ullah-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/sana-ullah-a799b22a8)
[![Kaggle](https://img.shields.io/badge/Kaggle-sanaullah03041417973-20BEFF?style=flat&logo=kaggle)](https://kaggle.com/sanaullah03041417973)

---

<div align="center">
  <sub>Developed by <strong>Sanaullah</strong> | Open Source</sub>
</div>

---

