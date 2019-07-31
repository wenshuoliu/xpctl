import re
from setuptools import setup, find_packages


def get_version(project_name):
    regex = re.compile(r"""^__version__ = ["'](\d+\.\d+\.\d+-?(?:a|b|rc|dev|alpha)?\.?(?:\d)*?)['"]$""")
    with open(f"{project_name}/version.py") as f:
        for line in f:
            m = regex.match(line.rstrip("\n"))
            if m is not None:
                return m.groups(1)[0]


class About(object):
    NAME = 'xpctl-sever'
    AUTHOR = 'mead-ml'
    VERSION = get_version('.')
    EMAIL = f"{AUTHOR}@gmail.com"


setup(
    name=About.NAME,
    version=About.VERSION,
    packages=find_packages(),
    install_requires=[
        'connexion',
        'pymongo',
        'sqlalchemy',
        'numpy'
    ],
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'xpctl-server = xpctl.xpserver.server:main',
        ]
    },
)
