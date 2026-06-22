# JetBot Research Repository

This repository is a starting point for research and development with the [SparkFun JetBot AI Kit v2.1 Powered by Jetson Nano](https://www.sparkfun.com/sparkfun-jetbot-ai-kit-v2-1-powered-by-jetson-nano.html).

The robot hardware is already assembled. The initial phase of this project is focused on basic robot control and system bring-up so each major component can be tested and verified. Later phases may explore more advanced autonomous control and research workflows.

## Current Focus

- Confirm the Jetson Nano boots reliably
- Verify motor control and basic driving behavior
- Check camera operation and image capture
- Confirm power, battery, and charging behavior
- Validate communication interfaces such as WiFi, SSH, and connected peripherals
- Document issues, fixes, and setup steps as the platform is stabilized

## Initial Goals

1. Establish a repeatable startup and shutdown process
2. Run simple manual or scripted motion tests
3. Verify that core hardware components work as expected
4. Create a clean baseline before beginning autonomy research

## Project Structure

- `scripts/` - Laptop-side Python tools for Vicon streaming, visualization, and teleoperation
- `notebooks/` - Jupyter notebooks used for bring-up tests and hardware experiments
- `chassis redesign/` - CAD files, exports, and reference parts for chassis modification work
- `images/` - Photos, diagrams, screenshots, and other project visuals

Additional folders for code, notes, experiments, and datasets can be added as the project grows.

## Bring-Up Checklist

- [ ] Power on the robot and confirm the Jetson Nano boots
- [ ] Verify remote access to the robot
- [ ] Test forward, reverse, left, and right motor commands
- [ ] Confirm camera feed is available
- [ ] Check any onboard display, indicators, and attached sensors
- [ ] Record any hardware or software issues discovered during testing

## Access Notes

The following workflow successfully brought the JetBot online and opened the basic motion notebook:

1. Connect a fully charged power bank to the JetBot through the micro USB power input and power the bot on.
2. Plug in a mouse, keyboard, and monitor to interact with the JetBot directly.
3. Connect both the JetBot and the laptop to a phone hotspot.
4. `utexas-iot` may require IT help before it can be used reliably. The MAC address may already be registered, or the device may be presenting a randomized MAC address.
5. Once the JetBot is online, open `10.53.174.144:8888` in a browser.
6. Log in with the current Jupyter password: `jetbot`.
7. In the Jupyter interface, click the folder icon near the top left.
8. Open the notebook file `basic_motion.ipynb`.

Result:
`basic_motion.ipynb` ran successfully, the notebook was self-explanatory, and basic motion control worked.

## Session Log

### 2026-06-22

- Added `scripts/vicon_goto_mppi_obstacle_controller.py` for MPPI-based target reaching that treats every other tracked Vicon object as a live obstacle.
- Visualized obstacle safety radii in the 3D room view so it is easier to see what the MPPI controller is trying to avoid.

### 2026-06-18

- Confirmed the Vicon UDP stream had about `1-2` seconds of latency on the laptop over Wi-Fi, while the same indicator reacted immediately on the Vicon machine and over wired Ethernet.
- Added `scripts/vicon_motion_indicator.py` as a minimal latency test that turns a large indicator red as soon as the tracked position moves past a threshold.
- Added `scripts/vicon_goto_continuous_viewer.py` for a continuously updated differential-drive go-to controller without the older step / stop / wait cadence.
- Added `scripts/vicon_goto_mppi_controller.py` for an MPPI-based point-to-point go-to controller using the same Vicon visualization and JetBot control link.
- Added `docs/continuous_goto_math.tex` and `docs/continuous_goto_math.pdf` to explain the math used by the continuous go-to controller.
- Updated `.gitignore` so generated LaTeX helper files in `docs/` do not appear as untracked changes.

### 2026-06-11

- Extended the Vicon visualization scripts to draw the room volume, robot footprint, and a horizontal heading arrow.
- Documented that the JetBot's physical forward direction matches the Vicon body-Y / green axis.
- Created `scripts/vicon_goto_viewer.py` for laptop-side target entry, Vicon visualization, and simple autonomous go-to control.
- Updated the go-to controller to use short differential-drive steps with pose-settle pauses instead of separate turn and drive phases.
- Added on-screen manual override buttons plus live controls for `epsilon`, heading offset, step time, and wait time.
- Set the default pose-settle wait in the go-to viewer to `1.0` second.

### 2026-06-10

- Connected the JetBot to the same LAN as the laptop and Vicon system using Ethernet.
- Found the JetBot on the network at `192.168.0.86`.
- Verified the JetBot web interface was reachable again over the shared local network.
- Confirmed the combined Vicon viewer and teleoperation script worked with the clickable control buttons.
- Added `chassis redesign/` to the repository for chassis CAD and export files.
- Added a `.gitignore` entry to ignore generated Python cache folders such as `__pycache__/`.

## Current Teleop Command

To run the combined Vicon viewer and clickable teleoperation controls from the laptop:

```bash
python scripts/vicon_teleop_viewer.py --jetbot-host 192.168.0.86 --source-ip 192.168.0.62 --object-name jetbot
```

Prerequisite:
Run `notebooks/jetbot_socket_server.ipynb` on the JetBot first so it can accept teleoperation commands from the laptop.

Current default teleoperation tuning in `scripts/vicon_teleop_viewer.py`:
- `speed = 0.7`
- `turn_speed = 0.5`

## Current Go-To Command

To run the Vicon room viewer with a simple go-to-target controller from the laptop:

```bash
python scripts/vicon_goto_viewer.py --jetbot-host 192.168.0.86 --source-ip 192.168.0.62 --object-name jetbot
```

Prerequisite:
Run `notebooks/jetbot_socket_server.ipynb` on the JetBot first so it can accept drive commands from the laptop.

Usage:
- Enter target `X` and `Y` coordinates in the GUI.
- Click `Go` to send the robot toward the target.
- The controller sends one short differential-drive step at a time, blending forward motion and steering, then stops, waits, checks the Vicon pose again, and repeats until it reaches the goal.
- Click `Stop Go-To` to cancel the current target.
- Press and hold the on-screen `Left`, `Forward`, `Stop`, `Right`, and `Reverse` buttons for manual override. Releasing a motion button stops the robot.
- Use `Epsilon` to control how close the robot must get before the target counts as reached.
- Use `Fwd Offset (deg)` to correct any fixed error in the heading direction that is treated as forward.
- Use `Step Time (s)` to control how long each differential-drive step lasts before stopping.
- Use `Wait Time (s)` to control how long the script waits after each step before checking the pose again. The current default is `1.0` second.
- The room view also shows the tracked room bounds, robot footprint, and a horizontal heading arrow based on the JetBot's Vicon green-axis forward direction.

## Current Continuous Go-To Command

To run the continuous differential-drive go-to controller from the laptop:

```bash
python scripts/vicon_goto_continuous_viewer.py --jetbot-host 192.168.0.86 --source-ip 192.168.0.62 --object-name jetbot
```

Prerequisite:
Run `notebooks/jetbot_socket_server.ipynb` on the JetBot first so it can accept drive commands from the laptop.

Usage:
- Enter target `X` and `Y` coordinates in the GUI.
- Click `Go` to drive continuously toward the target instead of moving in discrete pulses.
- The controller continuously recomputes the left and right wheel commands from the latest Vicon pose, target distance, and heading error.
- Click `Stop Go-To` to cancel the current target.
- Press and hold the on-screen manual drive buttons to override the controller at any time.
- Use `Epsilon` to set the arrival tolerance.
- Use `Angle Correction (deg)` to correct the heading vector used both for control and for the black heading arrow in the visualization.

## Current MPPI Go-To Command

To run the MPPI point-to-point controller from the laptop:

```bash
python scripts/vicon_goto_mppi_controller.py --jetbot-host 192.168.0.86 --source-ip 192.168.0.62 --object-name jetbot
```

Prerequisite:
Run `notebooks/jetbot_socket_server.ipynb` on the JetBot first so it can accept drive commands from the laptop.

Usage:
- Enter target `X` and `Y` coordinates in the GUI.
- Click `Go` to let the MPPI controller sample short left/right command sequences and choose a command that drives toward the target.
- This version is currently only for point-to-point target reaching. It does not include path following or obstacle costs yet.
- Click `Stop Go-To` to cancel the current target.
- Press and hold the manual drive buttons to override the controller at any time.
- Use `Epsilon` and `Angle Correction (deg)` the same way as in the continuous controller.

## Current MPPI Obstacle Command

To run the obstacle-aware MPPI controller from the laptop:

```bash
python scripts/vicon_goto_mppi_obstacle_controller.py --jetbot-host 192.168.0.86 --source-ip 192.168.0.62 --object-name jetbot
```

Prerequisite:
Run `notebooks/jetbot_socket_server.ipynb` on the JetBot first so it can accept drive commands from the laptop.

Usage:
- Enter target `X` and `Y` coordinates in the GUI.
- Click `Go` to let MPPI sample short wheel-command sequences and choose one that both approaches the target and avoids the other tracked Vicon objects.
- Every visible tracked object other than `jetbot` is treated as an obstacle automatically.
- The orange/red floor circles show the obstacle safety radius used by the rollout cost.
- Click `Stop Go-To` to cancel the current target.
- Press and hold the manual drive buttons to override the controller at any time.
- Use `Epsilon` and `Angle Correction (deg)` the same way as in the continuous and point-to-point MPPI controllers.
- Current obstacle-controller defaults are `Epsilon = 60 mm` and `Angle Correction = 40 deg`.
- Optional CLI tuning is available through `--obstacle-radius`, `--obstacle-influence-radius`, `--obstacle-cost-weight`, and `--obstacle-collision-cost`.

## Latency Test Command

To test whether the raw Vicon UDP stream is arriving late on the laptop:

```bash
python scripts/vicon_motion_indicator.py --source-ip 192.168.0.62 --object-name jetbot --threshold 2.0
```

Notes:
- The indicator turns red as soon as the tracked position moves more than the threshold from its armed baseline.
- Use `Re-arm Here` or press `R` after each test.
- This test showed the stream was delayed on Wi-Fi but responded immediately when the laptop was connected by wired Ethernet on the same network as the Vicon machine.

## Continuous Go-To Math Note

- LaTeX source: `docs/continuous_goto_math.tex`
- Compiled PDF: `docs/continuous_goto_math.pdf`
- The note explains the distance term, heading estimate, heading error, forward-speed schedule, steering term, and left/right wheel command equations used by `scripts/vicon_goto_continuous_viewer.py`.

### 2026-05-22

- Confirmed the onboard OLED status display appears to be part of the default JetBot software setup rather than code stored in this repository.
- Added `notebooks/basic_motion.ipynb` to this repository as the notebook used during bring-up testing.
- Created `notebooks/oled_hello_nathan.ipynb` as a simple OLED example that writes `Hello Nathan` to the screen.
- Tested the OLED notebook on the JetBot hardware and confirmed it worked.
- Documented that the default OLED status display can be restored by restarting `jetbot_stats.service`.
- Added a new session image: `images/20260522_164809.jpg`.
- Created `notebooks/camera_check.ipynb` to validate the onboard camera from Jupyter.
- Updated the camera notebook to use the stock `jetbot.Camera` import path after `jetcam` was not available on the device.
- Tested the camera notebook on the JetBot hardware and confirmed the onboard camera feed worked.
- Added a new camera test image: `images/2026-05-22 170202.png`.

## Research Direction

Near term work is centered on dependable low-level control and hardware validation. Once the platform is stable, this repository can expand to include:

- Autonomous navigation experiments
- Vision-based perception pipelines
- Sensor integration and state estimation
- Reinforcement learning or other advanced control approaches

## References

- SparkFun product page: <https://www.sparkfun.com/sparkfun-jetbot-ai-kit-v2-1-powered-by-jetson-nano.html>
- SparkFun documentation hub: <https://docs.sparkfun.com/>
