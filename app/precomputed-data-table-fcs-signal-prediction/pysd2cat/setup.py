from setuptools import setup, find_packages

setup(name='pysd2cat',
      version='0.1',
      description='Python Circuit Analysis Tool',
      url='http://gitlab.sd2e.org/pysd2cat',
      author='Dan Bryce',
      author_email='dbryce@sift.net',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'':'src'},
      install_requires=[
                      "pytest-runner",
                      "pymongo",
                      "pandas",
                      "scikit-learn",
                      "flowcytometrytools",
                      "pysbol>=2.3.3",
                      "pysmt"],
      tests_require=["pytest"],
      extras_require={
          "harness": ["test-harness@git+https://github.com/SD2E/test-harness.git@master"]},
      zip_safe=False)
