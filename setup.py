from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

# Include einstein.lark in src/einstein_tensors as well
if __name__ == "__main__":
    setup(
        name='einstein-tensors',
        version='0.0.1',
        description='Seriously terse Einstein summnation notation for tensors',
        author='Isaac Breen',
        author_email='mail@isaacbreen.com',
        url='https://github.com/IsaacBreen/EinsteinTensors',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'jax',
            'lark',
        ],
        include_package_data=True,
    )
