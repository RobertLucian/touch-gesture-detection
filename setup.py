import setuptools

with open('requirements.txt') as fp:
    install_requires = fp.read()

print(type(install_requires))

setuptools.setup(
    name='kiki',
    version='0.1.0',
    author='Robert Lucian Chiriac',
    author_email='robert.lucian.chiriac@gmail.com',
    description='A concept library for the Kiki AI robot',
    url='https://github.com/RobertLucian/touch-gesture-detection',
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering'
    ],
    include_package_data=True
)