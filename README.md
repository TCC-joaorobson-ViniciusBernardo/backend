# Backend

## Requirements

The requirements in this project are managed by [*pip-compile-multi*](https://pip-compile-multi.readthedocs.io/en/latest/). 
This tool avoids possible packages dependencies conflicts automating the control of the needed versions of the packages.

It works based on two files: *.in*, which contains the packages names and/or its versions and can be edited, and *.txt* files, 
automatically generated and that must be effectively used to install the dependencies.

Thus, if you need to add or remove a dependency, do it in the respective *.in* file, like this:

```
fastapi                                                                                             
uvicorn[standard] 
```

After that, in the root of the project, run:

```pip-compile-multi```

And the updated *.txt* file(s) will be generated.

### Requirements  classes

The requirements are dividided in *base* and *development* files. The first one is related to the essential dependencies needed to run the
project. The second contains libraries used only for development. It is recommended to create a virtual environment containing these development dependencies, so they can be used in the terminal. To do that, run:

```
python3.9 -m venv env
source env/bin/activate
pip install -r requirements/development.txt
```

## Running the API

To run the API, run:

```docker-compose up api```

The endpoints will be accessible at the 8000 port.

## Codestyle

Before submitting the code to the main branch, be sure to run the following commands:

```
pycodestyle --max-line-length=100 backend/
pylint backend/
```

They will show if thre are some conflict between the code and the main  Python patterns/conventions regarding
code quality.

It is also recommended to use as much as possible [black](https://github.com/psf/black). This tool
reduces the time doing hand-formatting and solves some problems that can be pointed by the tools above.
To run *black*, do the following:

```black --line-length 100 backend/```
