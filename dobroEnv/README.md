1. pip install pybullet
2. 해당 환경쓰려면 매 파일 상단에 아래 코드 넣기.

import subprocess
import sys
GITPATH = subprocess.run('git rev-parse --show-toplevel'.split(' '), \
        stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n','')
sys.path.append(GITPATH)
import dobroEnv
