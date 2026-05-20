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

## Research Direction

Near term work is centered on dependable low-level control and hardware validation. Once the platform is stable, this repository can expand to include:

- Autonomous navigation experiments
- Vision-based perception pipelines
- Sensor integration and state estimation
- Reinforcement learning or other advanced control approaches

## References

- SparkFun product page: <https://www.sparkfun.com/sparkfun-jetbot-ai-kit-v2-1-powered-by-jetson-nano.html>
- SparkFun documentation hub: <https://docs.sparkfun.com/>
