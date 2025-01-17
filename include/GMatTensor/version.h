/**
Version information.

\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATTENSOR_VERSION_H
#define GMATTENSOR_VERSION_H

#include "config.h"

/**
Current version.

Either:

-   Configure using CMake at install time. Internally uses::

        python -c "from setuptools_scm import get_version; print(get_version())"

-   Define externally using::

        -DGMATTENSOR_VERSION="`python -c "from setuptools_scm import get_version;
print(get_version())"`"

    From the root of this project. This is what ``setup.py`` does.

Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version using
``setuptools_scm``. Tip: use the environment variable ``SETUPTOOLS_SCM_PRETEND_VERSION`` to
overwrite the automatic version.
*/
#ifndef GMATTENSOR_VERSION
#define GMATTENSOR_VERSION "@PROJECT_VERSION@"
#endif

namespace GMatTensor {

/**
Return version string, e.g.::

    "0.8.0"

\return std::string
*/
inline std::string version();

} // namespace GMatTensor

#include "version.hpp"

#endif
