#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pip install GitPython
# pip install requests

import os
import argparse
import subprocess
from git import Repo
import requests


github_base_url = "https://api.github.com/repos/PaddlePaddle/Paddle/"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--repo_path",
    type=str,
    default='.',
    help="git repo path")
parser.add_argument(
    "--branch",
    type=str,
    default='',
    help="checkout repo branch name")
parser.add_argument(
    "--pr",
    type=str,
    default='',
    help="pull requests id")
parser.add_argument(
    "--commit",
    type=str,
    default='',
    help="commit id")
parser.add_argument(
    "--merge_before",
    type=str,
    default='false',
    help="pull code before merge, bool value, support true and false")


def _query_commit_info_from_github(commit_id):
    """
    # 查询 github 中指定 commit 相关的信息
    """
    url = github_base_url + "commits/" + str(commit_id)
    try:
        response = requests.get(url)
        parent_commit_id = response.json()["parents"][0]["sha"]
        print("parent of commit %s is commit %s" % (commit_id, parent_commit_id))
        result = {
            "parent_commit_id": parent_commit_id
        }
        return result
    except Exception as e:
        raise Exception("query commit info from github %s error: %s" % (
            url, str(e)
        ))


def _query_pull_request_info_from_github(pr_id):
    """
    # 查询 github 中指定 pr 相关的信息
    """
    url = github_base_url + "pulls/" + str(pr_id)
    try:
        response = requests.get(url)
        branch_name = response.json()["base"]["ref"]
        print("pr %s merge branch is %s" % (pr_id, branch_name))
        result = {
            "branch_name": branch_name
        }
        return result
    except Exception as e:
        raise Exception("query pr info from github %s error: %s" % (
            url, str(e)
        ))


def _parameters_check(args):
    """
    # parameter check
    """
    print("repo_path: %s" % args.repo_path)
    print("branch: %s" % args.branch)
    print("pr: %s" % args.pr)
    print("commit: %s" % args.commit)
    print("merge_before: %s" % args.merge_before)
    repo_path = os.path.abspath(args.repo_path)
    branch = args.branch
    pr = args.pr
    commit_id = args.commit
    merge_before = args.merge_before
    if merge_before.strip() not in ["true", "false"]:
        raise Exception("merge_before %s must be true or false" % str(repo_path))
    if not os.path.exists(repo_path):
        raise Exception("repo_path %s not exists" % str(repo_path))
    repo = Repo(repo_path)
    if repo.bare:
        raise Exception("repo_path %s is bare" % str(repo_path))
    if not branch and not pr and not commit_id:
        raise Exception("--branch, --pr, --commit is required at least one")
    if pr and commit_id:
        raise Exception("--pr, --commit is conflict, just can set one")
    if merge_before.strip() == "true" and not pr and not commit_id:
        raise Exception("when --merge_before is true, --pr or --commit is required at least one")


def _process_git_repo(args):
    """
    # process git repo, such as checkout branch and so on
    """
    # parameters check
    _parameters_check(args)

    repo_path = os.path.abspath(args.repo_path)
    branch = args.branch
    pr = args.pr
    commit_id = args.commit
    merge_before = args.merge_before
    if merge_before.strip() == "true":
        merge_before = True
    else:
        merge_before = False

    if not merge_before:
        # step1: 如果指定了branch，则checkout到对应branch
        if branch and branch != "develop" and not pr:
            commands = "cd %s && git checkout -b %s origin/%s" % (
                repo_path, branch, branch
            )
            print(commands)
            status, output = subprocess.getstatusoutput(commands)
            print(output)
            if status != 0:
                raise Exception("git checkout branch %s failed: %s" % (branch, output))

        # step2: 如果指定了pr,则拉取对应的pr,且判断是否有branch，如果有，则merge，如果没有，则checkout
        if pr:
            commands = "cd %s && git fetch origin pull/%s/head:%s" % (
                repo_path, str(pr), "pr-%s" % str(pr)
            )
            print(commands)
            status, output = subprocess.getstatusoutput(commands)
            print(output)
            if status != 0:
                raise Exception("git fetch pr %s failed: %s" % (str(pr), output))
            # query pr branch
            pr_info = _query_pull_request_info_from_github(pr)
            pr_into_branch = pr_info["branch_name"]
            commands = "cd %s && git checkout -b merge-pr-%s origin/%s" % (
                repo_path, pr, pr_into_branch
            )
            print(commands)
            status, output = subprocess.getstatusoutput(commands)
            print(output)
            if status != 0:
                raise Exception("git checkout branch %s failed: %s" % (pr_into_branch, output))

            commands = "cd %s && git merge %s" % (
                repo_path, "pr-%s" % str(pr)
            )
            print(commands)
            status, output = subprocess.getstatusoutput(commands)
            print(output)
            if status != 0:
                raise Exception("git merge pr %s failed: %s" % (str(pr), output))
            else:
                return 0

        # step3: 如果指定了commit id，则回滚到对应的commit id
        if commit_id:
            commands = "cd %s && git checkout -b %s %s" % (
                repo_path, "commit-%s" % commit_id, commit_id
            )
            print(commands)
            status, output = subprocess.getstatusoutput(commands)
            print(output)
            if status != 0:
                raise Exception("git checkout commit id %s failed: %s" % (str(pr), output))
            else:
                return 0
    else:
        # merge_before is true
        if pr:
            pr_info = _query_pull_request_info_from_github(pr)
            pr_into_branch = pr_info["branch_name"]
            if pr_into_branch != "develop":
                commands = "cd %s && git checkout -b %s origin/%s" % (
                    repo_path, pr_into_branch, branch
                )
                print(commands)
                status, output = subprocess.getstatusoutput(commands)
                print(output)
                if status != 0:
                    raise Exception("git checkout branch %s failed: %s" % (pr_into_branch, output))
                else:
                    return 0
            else:
                commands = "cd %s && git checkout develop" % (
                    repo_path
                )
                print(commands)
                status, output = subprocess.getstatusoutput(commands)
                print(output)
                if status != 0:
                    raise Exception("git checkout branch %s failed: %s" % (pr_into_branch, output))
                else:
                    return 0
        if commit_id:
            commit_info = _query_commit_info_from_github(commit_id)
            parent_commit_id = commit_info["parent_commit_id"]
            commands = "cd %s && git checkout -b %s %s" % (
                repo_path, "commit-%s" % parent_commit_id, parent_commit_id
            )
            print(commands)
            status, output = subprocess.getstatusoutput(commands)
            print(output)
            if status != 0:
                raise Exception("git checkout commit id %s failed: %s" % (str(pr), output))
            else:
                return 0


if __name__ == '__main__':
    # main
    args = parser.parse_args()
    _process_git_repo(args)
    # debug
    # _query_commit_info_from_github("2a771c06b7c7c89d39f2ba4c5bfd87768c533f65")
    # _query_pull_request_info_from_github(33065)
