from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()
        
setup(name='pymkm', 
      version='0.1',
      description='Microkinetic modeling framework for heterogeneous catalysis',
      classifiers=[
          'Development Status :: Experimental 0.1',
          'License :: MIT License',
          'Programming Language :: Python :: 3.8',
          'Topic :: Heterogeneous Catalysis :: Microkinetic Modeling',
      ],
      keywords='Heterogeneous Catalysis Electrocatalysis Microkinetic Modeling Computational Chemistry', 
      author='Santiago Morandi',
      author_email='smorandi@iciq.es',
      maintainer='Santiago Morandi',
      maintainer_email='smorandi@iciq.es',
      license='MIT License',
      packages=['pymkm'],
      install_requires=['numpy',
                        'pandas',
                        'matplotlib',
                        'scipy',
                        'natsort',
                        'graphviz',
                        ],
      include_package_data=True,
      zip_safe=False)
