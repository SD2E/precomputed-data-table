from setuptools import setup, find_packages

setup(name='precomputed_data_table',
      version='0.1',
      description='The Precomputed Data Table project for automating analyses using Data Converge products',
      url='https://gitlab.sd2e.org/sd2program/precomputed-data-table',
      author='Robert C. Moseley',
      author_email='robert.moseley@duke.edu',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'':'src'},
      zip_safe=False)