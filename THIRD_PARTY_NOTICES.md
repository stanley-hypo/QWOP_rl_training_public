# Third-Party Notices

This file lists third-party software and rights notices relevant to this
repository. The Apache License, Version 2.0 in the root LICENSE file applies only
to this repository's open-source code and documentation, not to separately
licensed third-party components or the proprietary precompiled physics backend.

## pybind11

The public Python extension uses [pybind11](https://github.com/pybind/pybind11),
licensed under a BSD-style license. The license is provided by the pybind11
submodule after running `git submodule update --init --recursive`.

## Precompiled physics backend

`bin/running_physics.dll` is a separately licensed proprietary binary component.
It is not licensed under the Apache License, Version 2.0. See
[`bin/running_physics.LICENSE.txt`](bin/running_physics.LICENSE.txt).

The precompiled physics backend may contain or be derived from software and
materials with their own rights and notices, including physics-engine and C
runtime components. Those notices apply only to the binary component and do not
change the license of this repository's open-source source code.

Known related notices for the precompiled backend:

- [Box2D](https://box2d.org/) - physics engine project. Check the upstream
  project for its current license terms.
- [musl libc](https://musl.libc.org/) - C library project. Check the upstream
  project for its current license terms.

## QWOP

QWOP is a game by Bennett Foddy. The QWOP name and related rights belong to
Bennett Foddy or their respective rights holder. This project is not affiliated
with or endorsed by the original game author. References to QWOP are descriptive
only.
