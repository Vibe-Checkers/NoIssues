"""
Improved Error Recognition and Docker Output Validation
Based on analysis of 18 failed repositories from empirical testing

This module provides:
1. Agent output validation and cleaning
2. Improved Docker error classification
3. Base image modernization
4. Automatic fixes for common issues
"""

import re
from typing import Tuple, Optional

class ImprovedErrorRecognition:
    """
    Enhanced error recognition system based on empirical findings.

    Key improvements:
    - Detects agent output format errors (DOCKERFILE_END in content, placeholders, etc.)
    - Better classification of Docker build errors
    - Distinguishes IMAGE_PULL from DEPENDENCY_BUILD_TOOLS
    - Detects when agent uses wrong syntax (shell operators in COPY)
    """

    @staticmethod
    def detect_output_format_errors(dockerfile_content: str) -> Tuple[bool, str]:
        """
        Detect when agent's output has formatting/parsing issues.

        Returns:
            (has_error, error_type) where error_type is one of:
            - "AGENT_OUTPUT_DELIMITER": Contains DOCKERFILE_END markers
            - "AGENT_OUTPUT_PLACEHOLDER": Contains [PATH], [HASH] placeholders
            - "COPY_SHELL_SYNTAX": COPY commands use shell syntax
            - "MISSING_FROM": No FROM statement
            - "" if no error
        """
        # Pattern 1: Delimiters in content
        if 'DOCKERFILE_END' in dockerfile_content or 'DOCKERFILE_START' in dockerfile_content:
            return (True, "AGENT_OUTPUT_DELIMITER")

        if 'DOCKERIGNORE_START' in dockerfile_content or 'DOCKERIGNORE_END' in dockerfile_content:
            return (True, "AGENT_OUTPUT_DELIMITER")

        # Pattern 2: Placeholder text not replaced
        if '[PATH]' in dockerfile_content or '[HASH]' in dockerfile_content:
            return (True, "AGENT_OUTPUT_PLACEHOLDER")

        if '[VERSION]' in dockerfile_content or '[TAG]' in dockerfile_content:
            return (True, "AGENT_OUTPUT_PLACEHOLDER")

        # Pattern 3: Shell syntax in COPY command
        copy_lines = [line for line in dockerfile_content.split('\n') if line.strip().startswith('COPY ')]
        for line in copy_lines:
            if '||' in line or '2>/dev/null' in line or '&>' in line:
                return (True, "COPY_SHELL_SYNTAX")

        # Pattern 4: Missing FROM statement
        if not re.search(r'^\s*FROM\s+', dockerfile_content, re.MULTILINE):
            return (True, "MISSING_FROM")

        return (False, "")

    @staticmethod
    def clean_agent_dockerfile_output(dockerfile_content: str) -> str:
        """
        Clean up agent output before writing to file.

        Removes:
        - Delimiter markers (DOCKERFILE_START, DOCKERFILE_END, etc.)
        - Placeholder text ([PATH], [HASH], etc.)
        - Shell syntax in COPY commands
        """
        original = dockerfile_content

        # Remove delimiter markers
        dockerfile_content = re.sub(r'DOCKERFILE_START\s*\n?', '', dockerfile_content)
        dockerfile_content = re.sub(r'\n?DOCKERFILE_END.*', '', dockerfile_content, flags=re.DOTALL)
        dockerfile_content = re.sub(r'DOCKERIGNORE_START.*DOCKERIGNORE_END', '', dockerfile_content, flags=re.DOTALL)

        # Remove placeholders - convert to sensible defaults
        # docker.io[PATH]/gcc:13@sha256:[HASH] → gcc:13
        dockerfile_content = re.sub(
            r'docker\.io\[PATH\]/([^:@\s]+):[^@\s]+@sha256:\[HASH\]',
            r'\1:latest',
            dockerfile_content
        )

        # [VERSION] → latest
        dockerfile_content = re.sub(r'\[VERSION\]', 'latest', dockerfile_content)
        dockerfile_content = re.sub(r'\[TAG\]', 'latest', dockerfile_content)

        # Fix COPY commands with shell syntax
        lines = dockerfile_content.split('\n')
        fixed_lines = []
        for line in lines:
            if line.strip().startswith('COPY '):
                # Remove shell operators
                line = re.sub(r'\s+2>/dev/null', '', line)
                line = re.sub(r'\s+\|\|\s+true', '', line)
                line = re.sub(r'\s+\|\|\s+false', '', line)
                line = re.sub(r'\s+&>', '', line)
                line = re.sub(r'\s+&&.*', '', line)  # Remove trailing &&
            fixed_lines.append(line)
        dockerfile_content = '\n'.join(fixed_lines)

        # Clean up any extra blank lines from removal
        dockerfile_content = re.sub(r'\n{3,}', '\n\n', dockerfile_content)

        return dockerfile_content.strip()

    @staticmethod
    def detect_old_base_image(dockerfile_content: str) -> Tuple[bool, str, str]:
        """
        Detect if Dockerfile uses deprecated/ancient base images.

        Returns:
            (is_old, original_image, suggested_replacement)
        """
        from_match = re.search(r'^\s*FROM\s+([^\s]+)', dockerfile_content, re.MULTILINE)
        if not from_match:
            return (False, "", "")

        image = from_match.group(1)

        # Ancient versions that definitely don't exist or are broken
        ANCIENT_PATTERNS = {
            r'python:2\.[0-9]': 'python:3.11-slim',
            r'python:3\.[0-7]([^0-9]|$)': 'python:3.10-slim',
            r'node:(0\.|[1-9]\.|1[0-7]\.)': 'node:20-slim',
            r'rust:1\.[0-5][0-9]': 'rust:1.75-slim',
            r'rust:1\.19': 'rust:1.75-slim',  # Specific case for deno
            r'maven:.*-jdk-[67]([^0-9]|$)': 'maven:3.9-eclipse-temurin-17',
            r'maven:3\.[0-7]': 'maven:3.9-eclipse-temurin-17',
            r'gcc:[1-9]\.[0-9]([^0-9]|$)': 'gcc:13',
            r'gcc:4\.': 'gcc:13',
            r'(debian|ubuntu):.*201[0-8]': r'\1:latest',
            r'openjdk:[67]-': 'eclipse-temurin:17-jdk-slim',
            r'openjdk:8-': 'eclipse-temurin:11-jdk-slim',
        }

        for pattern, replacement in ANCIENT_PATTERNS.items():
            if re.search(pattern, image):
                # Handle regex groups in replacement
                if r'\1' in replacement:
                    replacement = re.sub(pattern, replacement, image)
                return (True, image, replacement)

        return (False, "", "")

    @staticmethod
    def modernize_base_image(dockerfile_content: str) -> Tuple[str, bool]:
        """
        Automatically modernize old base images to current versions.

        Returns:
            (updated_dockerfile, was_modified)
        """
        is_old, old_image, new_image = ImprovedErrorRecognition.detect_old_base_image(dockerfile_content)

        if not is_old:
            return (dockerfile_content, False)

        # Replace the FROM line
        updated = re.sub(
            r'(^\s*FROM\s+)' + re.escape(old_image),
            r'\1' + new_image,
            dockerfile_content,
            flags=re.MULTILINE,
            count=1
        )

        return (updated, True)

    @staticmethod
    def prevent_manual_package_manager_install(dockerfile_content: str) -> Tuple[str, bool]:
        """
        Detect when agent tries to manually install package managers (Maven, Gradle)
        and replace with official base image.

        Returns:
            (updated_dockerfile, was_modified)
        """
        # Pattern: Manually installing Maven
        if 'curl' in dockerfile_content and 'apache-maven' in dockerfile_content:
            # Replace with maven base image
            updated = re.sub(
                r'FROM\s+[^\n]+',
                'FROM maven:3.9-eclipse-temurin-17 AS build',
                dockerfile_content,
                count=1
            )

            # Remove manual Maven installation lines
            lines = updated.split('\n')
            filtered = []
            skip_mode = False

            for line in lines:
                # Start skipping if we see Maven download
                if 'apache-maven' in line or ('curl' in line and 'maven' in line.lower()):
                    skip_mode = True
                    continue

                # Stop skipping when we hit a new Docker command
                if skip_mode:
                    if line.strip() and not line.strip().startswith('&&') and not line.strip().startswith('\\'):
                        # Check if this is a Docker instruction
                        if any(line.strip().upper().startswith(cmd) for cmd in ['FROM', 'RUN', 'COPY', 'ENV', 'WORKDIR', 'CMD', 'ENTRYPOINT', 'EXPOSE']):
                            skip_mode = False
                    else:
                        continue

                filtered.append(line)

            updated = '\n'.join(filtered)
            return (updated, True)

        # Pattern: Manually installing Gradle
        if 'curl' in dockerfile_content and 'gradle' in dockerfile_content.lower() and 'gradle.org' in dockerfile_content:
            updated = re.sub(
                r'FROM\s+[^\n]+',
                'FROM gradle:8-jdk17 AS build',
                dockerfile_content,
                count=1
            )

            # Remove Gradle installation
            lines = updated.split('\n')
            filtered = [line for line in lines if not ('gradle' in line.lower() and 'curl' in line)]
            updated = '\n'.join(filtered)
            return (updated, True)

        return (dockerfile_content, False)

    @staticmethod
    def classify_docker_build_error(error_log: str, failed_command: str, exit_code: int) -> str:
        """
        Improved error classification based on actual error patterns observed.

        Returns one of:
        - AGENT_OUTPUT_ERROR: Agent output format issues
        - IMAGE_PULL: Image doesn't exist or can't be pulled
        - EXTERNAL_DOWNLOAD_FAILED: curl/wget download failed
        - BUILD_FAILED: Actual compilation/build error
        - DEPENDENCY_MISSING_LIBRARY: Specific library needed (meson/cmake)
        - DEPENDENCY_BUILD_TOOLS: Missing gcc, make, etc.
        - DEPENDENCY_INSTALL: npm/pip/cargo install failed
        - DOCKERFILE_SYNTAX: Parse error in Dockerfile
        - FILE_COPY_MISSING: COPY file doesn't exist
        - UNKNOWN: Unclassified error
        """
        error_lower = error_log.lower()

        # PRIORITY 1: Agent output format errors (check first!)
        if 'unknown instruction: dockerfile_end' in error_lower:
            return 'AGENT_OUTPUT_ERROR'

        if 'unknown instruction: dockerignore' in error_lower:
            return 'AGENT_OUTPUT_ERROR'

        if 'invalid reference format' in error_lower and ('[' in error_log or 'PATH' in error_log):
            return 'AGENT_OUTPUT_ERROR'

        if '"/||": not found' in error_log:
            return 'AGENT_OUTPUT_ERROR'

        # PRIORITY 2: Image pull/registry errors
        image_pull_indicators = [
            'failed to resolve', 'manifest unknown', 'pull access denied',
            'image not found', 'no such image', 'not found: manifest',
            'error pulling image', 'failed to pull',
            'error downloading', 'failed to download',
            'unable to find image', 'repository does not exist',
            'received unexpected http status: 404',
            'received unexpected http status: 500',
            'received unexpected http status: 502',
            'toomanyrequests', 'rate limit'
        ]
        if any(x in error_lower for x in image_pull_indicators):
            return 'IMAGE_PULL'

        # PRIORITY 3: apt-get/apk exit code 100 = broken/old image
        if exit_code == 100 and 'apt-get' in failed_command.lower():
            return 'IMAGE_PULL'  # NOT DEPENDENCY_BUILD_TOOLS!

        if exit_code == 1 and 'apk' in failed_command.lower():
            if 'fetch' in error_lower or 'temporary error' in error_lower:
                return 'IMAGE_PULL'

        # PRIORITY 4: External download failures (curl, wget)
        if 'curl' in failed_command.lower() or 'wget' in failed_command.lower():
            if exit_code == 22:  # curl HTTP error
                return 'EXTERNAL_DOWNLOAD_FAILED'
            if any(x in error_lower for x in ['404', 'not found', 'failed to download', 'connection failed']):
                return 'EXTERNAL_DOWNLOAD_FAILED'

        # PRIORITY 5: Build command failures (NOT dependency install)
        build_commands = ['build', 'compile', 'make', 'ninja', 'gradle build', 'mvn package', 'pnpm run build', 'npm run build']
        if any(cmd in failed_command.lower() for cmd in build_commands):
            # Distinguish: missing tools vs build failure
            if any(x in error_lower for x in ['not found', 'command not found', 'no such file or directory']):
                return 'DEPENDENCY_BUILD_TOOLS'
            else:
                # Actual compilation/build error
                return 'BUILD_FAILED'

        # PRIORITY 6: Meson/CMake configuration failures (missing libraries)
        if any(x in failed_command.lower() for x in ['meson setup', 'meson configure', 'cmake']):
            if exit_code == 2 or 'dependency' in error_lower or 'not found' in error_lower:
                # meson exit code 2 = dependency not found
                return 'DEPENDENCY_MISSING_LIBRARY'

        # PRIORITY 7: Missing build tools
        missing_tool_indicators = [
            'gcc: not found', 'g++: not found', 'make: not found',
            'command not found: gcc', 'command not found: make',
            'unable to locate package build-essential',
            '/bin/sh: gcc: not found',
            '/bin/sh: make: not found',
        ]
        if any(x in error_lower for x in missing_tool_indicators):
            return 'DEPENDENCY_BUILD_TOOLS'

        # PRIORITY 8: Package manager failures (npm, pip, cargo, etc.)
        if any(x in error_lower for x in ['npm err!', 'npm error', 'yarn error']):
            # Distinguish missing system deps (gyp needs gcc) vs package errors
            if any(x in error_lower for x in ['gyp', 'node-gyp', 'gcc', 'compilation', 'prebuild']):
                return 'DEPENDENCY_BUILD_TOOLS'
            else:
                return 'DEPENDENCY_INSTALL'

        if 'pip install' in failed_command.lower() and 'failed' in error_lower:
            if any(x in error_lower for x in ['gcc', 'compilation', 'compiler']):
                return 'DEPENDENCY_BUILD_TOOLS'
            else:
                return 'DEPENDENCY_INSTALL'

        if 'cargo build' in failed_command.lower():
            if exit_code == 101:  # Cargo compilation error
                return 'BUILD_FAILED'
            else:
                return 'DEPENDENCY_INSTALL'

        # PRIORITY 9: File copy errors
        if ('no such file or directory' in error_lower or 'not found' in error_lower) and 'COPY' in failed_command:
            return 'FILE_COPY_MISSING'

        # PRIORITY 10: Syntax errors
        if any(x in error_lower for x in ['syntax error', 'parse error', 'unknown instruction', 'dockerfile parse error']):
            return 'DOCKERFILE_SYNTAX'

        # Default
        return 'UNKNOWN'


# Example usage
if __name__ == "__main__":
    recognizer = ImprovedErrorRecognition()

    # Test case 1: FreeRTOS - delimiter in content
    dockerfile_freertos = """FROM ubuntu:24.04
RUN apt-get update
CMD ["/bin/bash"]
DOCKERFILE_END

DOCKERIGNORE_START
.git
DOCKERIGNORE_END
"""
    has_error, error_type = recognizer.detect_output_format_errors(dockerfile_freertos)
    print(f"FreeRTOS: has_error={has_error}, type={error_type}")

    cleaned = recognizer.clean_agent_dockerfile_output(dockerfile_freertos)
    print(f"Cleaned:\n{cleaned}\n")

    # Test case 2: msgpack-c - placeholder
    dockerfile_msgpack = "FROM docker.io[PATH]/gcc:13@sha256:[HASH]\nWORKDIR /app"
    has_error, error_type = recognizer.detect_output_format_errors(dockerfile_msgpack)
    print(f"msgpack-c: has_error={has_error}, type={error_type}")

    cleaned = recognizer.clean_agent_dockerfile_output(dockerfile_msgpack)
    print(f"Cleaned:\n{cleaned}\n")

    # Test case 3: Old base image detection
    dockerfile_old = "FROM python:2.7.9\nRUN pip install numpy"
    is_old, old_img, new_img = recognizer.detect_old_base_image(dockerfile_old)
    print(f"Old image: is_old={is_old}, {old_img} → {new_img}")

    modernized, was_modified = recognizer.modernize_base_image(dockerfile_old)
    print(f"Modernized:\n{modernized}\n")

    # Test case 4: Error classification
    error_log_apt = "ERROR: exit code: 100\napt-get update failed"
    classification = recognizer.classify_docker_build_error(error_log_apt, "RUN apt-get update", 100)
    print(f"apt-get exit 100: {classification}")

    error_log_delimiter = "ERROR: unknown instruction: DOCKERFILE_END"
    classification = recognizer.classify_docker_build_error(error_log_delimiter, "FROM ubuntu", 1)
    print(f"Delimiter error: {classification}")

    error_log_curl = "ERROR: exit code: 22\ncurl failed"
    classification = recognizer.classify_docker_build_error(error_log_curl, "RUN curl https://example.com/maven.tar.gz", 22)
    print(f"Curl error: {classification}")
