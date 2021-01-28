Installation
============

Install ALLib
-------------

The ALLib library requires having Python 3.8 or higher installed. 
Install the library with `pip` by running the following command in the
`CMD.exe` (Windows) or `Terminal` (MacOS/Linux):

.. code:: bash

    pip install allib

You are now ready to start your first Active Learning experiments!

See `Troubleshooting`_ 



Upgrade ALLib
-------------

Upgrade ALLib with

.. code:: bash

    pip install --upgrade allib



Uninstall ALLib
------------------

Remove ALLib with

.. code:: bash

    pip uninstall allib

Enter ``y`` to confirm. 



Troubleshooting
---------------

ALLib is an advanced machine learning library. In some situations, you
might run into unexpected behavior. See below for solutions to
problems.

Unknown Command "pip"
~~~~~~~~~~~~~~~~~~~~~

The command line returns one of the following messages:

.. code:: bash

  -bash: pip: No such file or directory

.. code:: bash

  'pip' is not recognized as an internal or external command, operable program or batch file.


First, check if Python is installed with the following command:

.. code:: bash

    python --version

If this doesn't return 3.8 or higher, then Python isn't or not correctly
installed.

However, there is a simple way to deal with correct environment variables
by ading `python -m` in front of the command. For example:

.. code:: bash

  python -m pip install allib
