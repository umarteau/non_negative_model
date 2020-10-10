import os
from setuptools import setup, find_packages


def get_requirements(filename='requirements.txt'):
    ret = []
    if os.path.isfile(filename):
        for x in open(filename):
            ret.append(x.strip())
    return ret


def get_version(dir_name):
    print(os.getcwd())
    for x in open(os.path.join(dir_name, 'npm_nnf/utils/__init__.py')):
        x = [y.strip() for y in x.split('=')]
        if len(x) == 2 and x[0] == '__version__':
            return eval(x[1])
    raise ValueError('__version__ not found in __init__.py')


def get_scripts(dir_name):
    ret = []
    if os.path.isdir(dir_name):
        for fn in os.listdir(dir_name):
            ret.append(os.path.join(dir_name, fn))
    return ret


def get_data(dir_name):
    ret = []
    if os.path.isdir(dir_name):
        for d, a, l in os.walk(dir_name):
            ret.append((d, [os.path.join(d, x) for x in l]))
    return ret


setup(
    name='npm_nnf',
    version = '0.0.1',
    author='Ulysse',
    author_email='ulysse.marteau-ferey@inria.fr',
    maintainer='ulysse',
    maintainer_email='ulysse.marteau-ferey@inria.fr',
    packages=find_packages('.'),
    description='haha',
    long_description='hihi',
    install_requires=get_requirements(),
    #scripts=get_scripts('analysts_reports/scripts')
)