from setuptools import setup, find_packages

setup(
    name='my_project',  # The name of your project
    version='0.1.0',  # The initial release version
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[  # External dependencies your project requires
        # List dependencies here, for example:
        # 'requests',
        # 'numpy',
    ],
    entry_points={  # Entry points for command-line scripts, if any
        'console_scripts': [
            # 'my_command=my_module:main_function',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of your project',
    url='https://github.com/yourusername/yourproject',  # Your project's GitHub or website
)