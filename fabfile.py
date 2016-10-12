import fabric.contrib.project as project
from fabric.api import local, settings, abort, run, cd, env, task
from fabric.context_managers import *
import os
env.use_ssh_config = True

#env.hosts = ["teslabot", "gpunode2"]
hosts_config = {
        "teslabot":"/home/eror/",
        "gpunode1":"/home/przymusp/",
        "gpunode2":"/home/samba/przymusp/",
        "gpunode3":"/home/samba/przymusp/",
        "cadmed":"/home/samba/przymusp/",
        #gpunode2 z PW
        "192.168.137.7":"/home/samba/przymusp/",
        "192.168.137.6":"/home/samba/przymusp/",
        }

@task
def sync():
    exclude_list=['benchmarks*', 'real_data_benchmarks*']
    project.rsync_project(remote_dir=hosts_config[env.host], exclude=exclude_list, default_opts="-pthrz")

@task
def make(*args):
    with settings(output_prefix=False), cd(hosts_config[env.host] + "/" + os.getcwd().split("/")[-1]):
        run("pwd")
        for arg in args:
            run("make -j 12 " + arg)

@task(default=True)
def all(*args):
    sync()
    make(*args)
    local("make ctags")
