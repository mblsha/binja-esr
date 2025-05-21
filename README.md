# Binary Ninja ESR-* plugin

The ESR plugin provides an SC62015 (aka ESR-L) architecture for Binary Ninja.

Currently it only works as a crude disassembler, with the goal to lift all the
instructions and create memory mapping for Sharp PC-E500 and Sharp Organizers.

## Acknowledgements

Overall structure of instruction logic based on
[binja-avnera](https://github.com/whitequark/binja-avnera) plugin by
@whitequark.

## License

Apache License 2.0.

## Testing the CI Workflow Locally

To test the GitHub Actions CI workflow (`.github/workflows/ci.yml`) locally before pushing changes, you can use [act](https://github.com/nektos/act). This tool runs your workflows in a local Docker environment.

### Prerequisites

1.  **Docker**: Ensure Docker is installed and running on your system.
2.  **act**: Install `act`. For example, using Homebrew on macOS or Linux:
    ```bash
    brew install act
    ```
    For other installation methods, refer to the [act installation guide](https://github.com/nektos/act#installation).

### Running the Workflow

Navigate to the root directory of this repository in your terminal and run the following commands:

*   To simulate a `push` event (which is the default event `act` runs):
    ```bash
    act
    ```
*   To simulate a `pull_request` event:
    ```bash
    act pull_request
    ```
*   To run a specific job within the workflow (e.g., `build_and_test`):
    ```bash
    act -j build_and_test
    ```

### Notes

*   `act` will download the necessary Docker images to mimic the GitHub Actions runners, which might take some time on the first run.
*   If your workflow uses secrets (not currently the case for `ci.yml`), you might need to provide them to `act` using a `.secrets` file or environment variables. See the `act` documentation for more details.
*   For private repositories, `act` might require a `GITHUB_TOKEN` with appropriate permissions.
