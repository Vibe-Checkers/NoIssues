#!/usr/bin/env python3
"""
Planner Agent for Automated Project Building
Uses LangChain to analyze GitHub repositories and create installation guides
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProjectInfo:
    """Container for project information"""
    repo_owner: str
    repo_name: str
    repo_url: str
    stars: Optional[int] = None
    language: Optional[str] = None
    description: Optional[str] = None
    build_system: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class InstallationGuide:
    """Container for generated installation guide"""
    project_info: ProjectInfo
    prerequisites: List[str]
    installation_steps: List[Dict[str, Any]]
    build_commands: List[str]
    verification_steps: List[str]
    troubleshooting: List[str]
    references: List[str]
    generated_at: str


class PlannerAgent:
    """
    Planner agent that analyzes GitHub repositories and creates
    detailed installation guides for builders.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        openai_api_key: Optional[str] = None,
        use_azure: bool = False
    ):
        """
        Initialize the planner agent
        
        Args:
            model_name: OpenAI model to use
            temperature: Model temperature
            openai_api_key: OpenAI API key (optional, can also use OPENAI_API_KEY or AZURE_OPENAI_API_KEY env var)
            use_azure: Whether to use Azure OpenAI
        """
        # Check if we should use Azure OpenAI
        if use_azure or os.getenv("AZURE_OPENAI_API_KEY"):
            self.use_azure = True
            
            # Get Azure credentials - try hardcoded first, then environment
            self.azure_api_key = openai_api_key or os.getenv("AZURE_OPENAI_API_KEY")
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            
            if not self.azure_api_key:
                raise ValueError("Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY env var or pass as argument.")
            if not self.azure_endpoint:
                raise ValueError("Azure OpenAI endpoint not provided. Set AZURE_OPENAI_ENDPOINT env var.")
            if not self.azure_deployment:
                raise ValueError("Azure OpenAI deployment not provided. Set AZURE_OPENAI_DEPLOYMENT env var.")
            
            # Initialize Azure OpenAI LLM
            self.llm = AzureChatOpenAI(
                azure_deployment=self.azure_deployment,
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version="2024-02-15-preview",
                temperature=temperature
            )
            
            logger.info("Initialized with Azure OpenAI")
        else:
            # Use standard OpenAI
            self.use_azure = False
            self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass as argument.")
            
            # Set the API key in environment for LangChain
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature
            )
            
            logger.info("Initialized with standard OpenAI")
        
        # Initialize web search tool
        self.search = DuckDuckGoSearchRun()
        
        # Language-specific build knowledge
        self.language_knowledge = {
            "Python": {
                "build_tools": ["pip", "poetry", "conda", "setuptools"],
                "config_files": ["setup.py", "pyproject.toml", "requirements.txt", "Pipfile"],
                "common_commands": ["python setup.py install", "pip install -e .", "poetry install"]
            },
            "JavaScript": {
                "build_tools": ["npm", "yarn", "pnpm", "bun"],
                "config_files": ["package.json", "yarn.lock", "pnpm-lock.yaml"],
                "common_commands": ["npm install", "npm run build", "yarn install"]
            },
            "Go": {
                "build_tools": ["go", "make"],
                "config_files": ["go.mod", "go.sum", "Makefile", "Gopkg.toml"],
                "common_commands": ["go mod download", "go build", "make install"]
            },
            "Rust": {
                "build_tools": ["cargo", "rustc"],
                "config_files": ["Cargo.toml", "Cargo.lock"],
                "common_commands": ["cargo build", "cargo install", "rustup"]
            },
            "Java": {
                "build_tools": ["maven", "gradle", "ant"],
                "config_files": ["pom.xml", "build.gradle", "build.xml"],
                "common_commands": ["mvn install", "gradle build", "./gradlew build"]
            },
            "C++": {
                "build_tools": ["cmake", "make", "bazel", "ninja"],
                "config_files": ["CMakeLists.txt", "Makefile", "BUILD", ".bazelrc"],
                "common_commands": ["cmake . && make", "bazel build", "make install"]
            },
            "C": {
                "build_tools": ["make", "cmake", "gcc", "clang"],
                "config_files": ["Makefile", "CMakeLists.txt"],
                "common_commands": ["make", "make install", "cmake . && make"]
            }
        }
    
    def fetch_repo_info(self, repo_url: str) -> ProjectInfo:
        """
        Fetch repository information from GitHub
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            ProjectInfo object with repository details
        """
        try:
            # Parse GitHub URL
            if "github.com" not in repo_url:
                raise ValueError(f"Invalid GitHub URL: {repo_url}")
            
            parts = repo_url.replace("https://github.com/", "").replace(".git", "").split("/")
            if len(parts) < 2:
                raise ValueError(f"Invalid GitHub URL format: {repo_url}")
            
            owner, repo = parts[0], parts[1]
            
            # Fetch from GitHub API
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            logger.info(f"Fetching repository info from: {api_url}")
            
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            
            # Determine primary language
            language = data.get("language", "Unknown")
            
            return ProjectInfo(
                repo_owner=owner,
                repo_name=repo,
                repo_url=repo_url,
                stars=data.get("stargazers_count", 0),
                language=language,
                description=data.get("description", ""),
                build_system=None,  # Will be determined later
                domain=None
            )
            
        except Exception as e:
            logger.error(f"Error fetching repo info: {e}")
            raise
    
    def fetch_repo_contents(self, project_info: ProjectInfo) -> Dict[str, Any]:
        """
        Fetch repository contents using GitHub API
        
        Args:
            project_info: ProjectInfo object
            
        Returns:
            Dictionary with repository contents
        """
        try:
            api_url = f"https://api.github.com/repos/{project_info.repo_owner}/{project_info.repo_name}/contents"
            logger.info(f"Fetching repository contents from: {api_url}")
            
            response = requests.get(api_url)
            response.raise_for_status()
            contents = response.json()
            
            # Extract file names
            files = {}
            for item in contents:
                if item["type"] == "file":
                    files[item["name"]] = item.get("download_url", "")
            
            return {"files": files, "raw": contents}
            
        except Exception as e:
            logger.error(f"Error fetching repo contents: {e}")
            return {"files": {}, "raw": []}
    
    def fetch_file_content(self, project_info: ProjectInfo, filename: str) -> Optional[str]:
        """
        Fetch specific file content from repository
        
        Args:
            project_info: ProjectInfo object
            filename: Name of file to fetch
            
        Returns:
            File contents as string or None
        """
        try:
            api_url = f"https://api.github.com/repos/{project_info.repo_owner}/{project_info.repo_name}/contents/{filename}"
            logger.info(f"Fetching file: {filename}")
            
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            
            import base64
            content = base64.b64decode(data["content"]).decode("utf-8")
            return content
            
        except Exception as e:
            logger.warning(f"Could not fetch file {filename}: {e}")
            return None
    
    def search_language_docs(self, language: str, project_info: ProjectInfo) -> List[str]:
        """
        Search for language-specific documentation and best practices
        
        Args:
            language: Programming language
            project_info: Project information
            
        Returns:
            List of relevant documentation snippets
        """
        if language not in self.language_knowledge:
            logger.warning(f"Unknown language: {language}")
            return []
        
        knowledge = self.language_knowledge[language]
        results = []
        
        # Web search for language-specific installation guides
        search_queries = [
            f"{language} {project_info.repo_name} installation guide",
            f"{language} {knowledge['build_tools'][0]} best practices",
            f"how to build {project_info.repo_name} from source"
        ]
        
        for query in search_queries:
            try:
                search_result = self.search.run(query)
                results.append(f"Search: {query}\n{search_result}")
                logger.info(f"Search results for: {query}")
            except Exception as e:
                logger.warning(f"Search failed for {query}: {e}")
        
        return results
    
    def analyze_repo_structure(self, project_info: ProjectInfo) -> Dict[str, Any]:
        """
        Analyze repository structure to determine build system
        
        Args:
            project_info: Project information
            
        Returns:
            Dictionary with build system analysis
        """
        contents = self.fetch_repo_contents(project_info)
        files = contents.get("files", {})
        
        # Identify build configuration files
        detected_configs = []
        language = project_info.language
        
        if language in self.language_knowledge:
            config_files = self.language_knowledge[language]["config_files"]
            for config_file in config_files:
                if config_file in files:
                    detected_configs.append(config_file)
                    logger.info(f"Detected config file: {config_file}")
        
        # Common files to check
        common_files = ["README.md", "INSTALL.md", "BUILD.md", "CONTRIBUTING.md", "Makefile", "build.sh"]
        found_files = {f: files.get(f) for f in common_files if f in files}
        
        return {
            "detected_configs": detected_configs,
            "found_files": found_files,
            "language": language
        }
    
    def read_repository_docs(self, project_info: ProjectInfo) -> str:
        """
        Read README and other documentation from repository
        
        Args:
            project_info: Project information
            
        Returns:
            Concatenated documentation content
        """
        docs = []
        
        # Common documentation files
        doc_files = ["README.md", "INSTALL.md", "BUILD.md", "CONTRIBUTING.md", "requirements.txt", 
                     "package.json", "setup.py", "pyproject.toml", "CMakeLists.txt", "Makefile"]
        
        for filename in doc_files:
            content = self.fetch_file_content(project_info, filename)
            if content:
                docs.append(f"=== {filename} ===\n{content}\n")
                logger.info(f"Read documentation from: {filename}")
        
        return "\n".join(docs)
    
    def generate_installation_guide(self, repo_url: str) -> InstallationGuide:
        """
        Main method to generate installation guide for a repository
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            InstallationGuide object
        """
        logger.info(f"Generating installation guide for: {repo_url}")
        
        # Step 1: Fetch repository information
        project_info = self.fetch_repo_info(repo_url)
        logger.info(f"Project: {project_info.repo_owner}/{project_info.repo_name}")
        logger.info(f"Language: {project_info.language}")
        
        # Step 2: Read repository documentation
        repo_docs = self.read_repository_docs(project_info)
        logger.info("Read repository documentation")
        
        # Step 3: Analyze repository structure
        repo_analysis = self.analyze_repo_structure(project_info)
        logger.info(f"Detected {len(repo_analysis['detected_configs'])} config files")
        
        # Step 4: Search for language-specific documentation
        language_docs = []
        if project_info.language:
            language_docs = self.search_language_docs(project_info.language, project_info)
            logger.info("Searched language-specific documentation")
        
        # Step 5: Generate installation guide using LLM
        guide = self._generate_guide_with_llm(
            project_info,
            repo_docs,
            repo_analysis,
            language_docs
        )
        
        logger.info("Installation guide generated successfully")
        return guide
    
    def _generate_guide_with_llm(
        self,
        project_info: ProjectInfo,
        repo_docs: str,
        repo_analysis: Dict[str, Any],
        language_docs: List[str]
    ) -> InstallationGuide:
        """
        Use LLM to generate structured installation guide
        
        Args:
            project_info: Project information
            repo_docs: Repository documentation
            repo_analysis: Repository analysis
            language_docs: Language-specific documentation
            
        Returns:
            InstallationGuide object
        """
        
        # Create prompt for LLM
        prompt_template = """You are an expert at creating detailed installation and build guides for open-source projects.
You will be given information about a GitHub repository and need to create a comprehensive installation guide.

PROJECT INFORMATION:
Owner: {owner}
Name: {name}
Language: {language}
Stars: {stars}
Description: {description}

REPOSITORY DOCUMENTATION:
{docs}

DETECTED CONFIGURATION FILES:
{configs}

LANGUAGE-SPECIFIC KNOWLEDGE:
{lang_docs}

INSTRUCTIONS:
Based on the information provided, create a detailed installation guide with:

1. PREREQUISITES: List all required software, tools, and dependencies
2. INSTALLATION STEPS: Step-by-step instructions
3. BUILD COMMANDS: Specific commands to build/install the project
4. VERIFICATION: How to verify the installation worked
5. TROUBLESHOOTING: Common issues and solutions
6. REFERENCES: Links to relevant documentation

Be specific, clear, and practical. Format the output as structured JSON that can be parsed.
"""

        language_info = ""
        if project_info.language and project_info.language in self.language_knowledge:
            lang_knowledge = self.language_knowledge[project_info.language]
            language_info = f"""
Build Tools: {', '.join(lang_knowledge['build_tools'])}
Config Files: {', '.join(lang_knowledge['config_files'])}
Common Commands: {chr(10).join(['- ' + cmd for cmd in lang_knowledge['common_commands']])}
"""
        
        prompt = prompt_template.format(
            owner=project_info.repo_owner,
            name=project_info.repo_name,
            language=project_info.language or "Unknown",
            stars=project_info.stars or 0,
            description=project_info.description or "No description",
            docs=repo_docs[:5000] if repo_docs else "No documentation found",
            configs="\n".join(repo_analysis.get("detected_configs", [])),
            lang_docs=language_info
        )
        
        # Call LLM
        messages = [
            SystemMessage(content="You are a helpful assistant that creates detailed, practical installation guides for software projects."),
            HumanMessage(content=prompt)
        ]
        
        logger.info("Calling LLM to generate installation guide...")
        response = self.llm.invoke(messages)
        guide_text = response.content
        
        # Parse guide into structured format
        # For now, create a structured guide from the LLM output
        # In production, you might want better parsing or use structured outputs
        guide = self._parse_llm_response(guide_text, project_info)
        
        return guide
    
    def _parse_llm_response(self, response: str, project_info: ProjectInfo) -> InstallationGuide:
        """
        Parse LLM response into structured InstallationGuide
        
        Args:
            response: LLM response text
            project_info: Project information
            
        Returns:
            InstallationGuide object
        """
        # Simple parsing - in production, use more sophisticated parsing
        # or LLM structured output
        
        prerequisites = []
        steps = []
        commands = []
        verification = []
        troubleshooting = []
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if "PREREQUISITE" in line.upper() or "REQUIREMENT" in line.upper():
                current_section = "prerequisites"
                continue
            elif "STEP" in line.upper() or "INSTALL" in line.upper():
                current_section = "steps"
                continue
            elif "COMMAND" in line.upper() or "BUILD" in line.upper():
                current_section = "commands"
                continue
            elif "VERIFY" in line.upper() or "TEST" in line.upper():
                current_section = "verification"
                continue
            elif "TROUBLESHOOT" in line.upper() or "ISSUE" in line.upper():
                current_section = "troubleshooting"
                continue
            
            # Parse content
            if current_section == "prerequisites" and line.startswith("-"):
                prerequisites.append(line[1:].strip())
            elif current_section == "steps" and (line[0].isdigit() or line.startswith("-")):
                steps.append({"step": line, "command": ""})
            elif current_section == "commands" and ("`" in line or "$" in line or ">" in line):
                commands.append(line)
            elif current_section == "verification" and line.startswith("-"):
                verification.append(line[1:].strip())
            elif current_section == "troubleshooting" and line.startswith("-"):
                troubleshooting.append(line[1:].strip())
        
        # Ensure we have some default content
        if not prerequisites:
            prerequisites = ["Git", "Required programming language and build tools"]
        if not steps:
            steps = [{"step": "Clone the repository", "command": f"git clone {project_info.repo_url}"}]
        if not commands:
            commands = [f"cd {project_info.repo_name}"]
        
        return InstallationGuide(
            project_info=project_info,
            prerequisites=prerequisites,
            installation_steps=steps,
            build_commands=commands,
            verification_steps=verification if verification else ["Run test suite if available"],
            troubleshooting=troubleshooting,
            references=[project_info.repo_url],
            generated_at=datetime.now().isoformat()
        )


def main():
    """Example usage of the PlannerAgent"""
    
    # Example repositories
    test_repos = [
        "https://github.com/psf/requests",
        "https://github.com/microsoft/playwright",
        "https://github.com/kubernetes/kubernetes"
    ]
    
    # Initialize planner agent
    try:
        planner = PlannerAgent()
        
        # Generate installation guide for first repo
        repo_url = test_repos[0]
        logger.info(f"Generating guide for: {repo_url}")
        
        guide = planner.generate_installation_guide(repo_url)
        
        # Print guide
        print("\n" + "="*70)
        print("INSTALLATION GUIDE")
        print("="*70)
        print(f"\nProject: {guide.project_info.repo_owner}/{guide.project_info.repo_name}")
        print(f"Language: {guide.project_info.language}")
        
        print("\nPrerequisites:")
        for prereq in guide.prerequisites:
            print(f"  - {prereq}")
        
        print("\nInstallation Steps:")
        for i, step in enumerate(guide.installation_steps, 1):
            print(f"  {i}. {step['step']}")
            if step.get('command'):
                print(f"     {step['command']}")
        
        print("\nBuild Commands:")
        for cmd in guide.build_commands:
            print(f"  {cmd}")
        
        print("\nVerification:")
        for verify in guide.verification_steps:
            print(f"  - {verify}")
        
        print("\nTroubleshooting:")
        for trouble in guide.troubleshooting:
            print(f"  - {trouble}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
        print("\nMake sure to set OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    main()

