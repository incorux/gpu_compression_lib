import fabric.contrib.project as project
from fabric.api import local, settings, abort, run, cd, env, task
from fabric.context_managers import *
env.use_ssh_config = True

#env.hosts = ["teslabot", "gpunode2"]
hosts_config = {
        "teslabot":"/home/eror/",
        "gpunode2":"/home/samba/przymusp/",
        }

@task
def sync():
    project.rsync_project(remote_dir=hosts_config[env.host], extra_opts="--filter=':- .gitignore'", default_opts="-pthrz")

@task
def make(*args):
    with settings(output_prefix=False), cd(hosts_config[env.host]):
        run("pwd")
        for arg in args:
            run("make -j 12 " + arg)

@task(default=True)
def all(*args):
    sync()
    make(*args)
    local("make ctags")
