
import sys
import unittest
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup

# Add src to path
sys.path.append('src')

from agent.validation import DockerBuildTester
from agent.validation import DockerBuildTester
from parallel_empirical_test import LLMErrorAnalyzer
from agent.tools import extract_relevant_sections
from agent.tools import extract_relevant_sections

class TestReliabilityFixes(unittest.TestCase):
    
    @patch('subprocess.run')
    @patch('time.sleep') # Skip sleep
    def test_docker_retry_logic(self, mock_run, mock_sleep):
        print("\nTesting DockerBuildTester Retry Logic...")
        tester = DockerBuildTester(timeout=10)
        
        # Mock first failure (transient), second success
        failure_result = MagicMock()
        failure_result.returncode = 1
        failure_result.stdout = "some output"
        failure_result.stderr = "E: Failed to fetch http://deb.debian.org/... Connection timed out"
        
        success_result = MagicMock()
        success_result.returncode = 0
        success_result.stdout = "Success"
        success_result.stderr = ""
        
        # 1. First call: _check_docker (during init) -> Success
        # 2. Second call: build (failed)
        # 3. Third call: build (retry success)
        
        check_docker_result = MagicMock()
        check_docker_result.returncode = 0
        
        mock_run.side_effect = [check_docker_result, failure_result, success_result]
        
        # Create a dummy dockerfile for the check
        with patch('os.path.exists', return_value=True):
             result = tester.build_dockerfile("DummyDockerfile", ".", "test-image")
        
        print(f"DEBUG: Result: {result}")
        print(f"DEBUG: Mock calls: {len(mock_run.mock_calls)}")
        for call in mock_run.mock_calls:
            print(f"DEBUG: Call: {call}")

        self.assertTrue(result['success'], "Build should succeed after retry")
        self.assertTrue(result.get('was_retry', False), "Result should indicate it was a retry")
        print("✓ Retry logic verified!")

    @patch('parallel_empirical_test.AzureChatOpenAI')
    def test_llm_error_analyzer(self, mock_azure_openai):
        print("\nTesting LLMErrorAnalyzer...")
        
        # Mock LLM response
        mock_llm_instance = mock_azure_openai.return_value
        mock_response = MagicMock()
        mock_response.content = """
        ```json
        {
            "cause": "Missing development headers for Python",
            "missing_packages": ["python3-dev"],
            "suggested_fix": "RUN apt-get install -y python3-dev",
            "search_keywords": "python.h not found docker"
        }
        ```
        """
        mock_llm_instance.invoke.return_value = mock_response
        
        analyzer = LLMErrorAnalyzer()
        log = "fatal error: Python.h: No such file or directory"
        analysis = analyzer.analyze_error(log, "pip install numpy")
        
        print(f"Analysis result: {analysis}")
        
        self.assertEqual(analysis['cause'], "Missing development headers for Python")
        self.assertIn("python3-dev", analysis['missing_packages'])
        self.assertEqual(analysis['suggested_fix'], "RUN apt-get install -y python3-dev")
        print("✓ LLM Analyzer verified!")

    def test_web_search_optimization(self):
        print("\nTesting Web Search Optimization...")
        
        html = """
        <html><body>
        <nav>Menu</nav>
        <p>Some random text about the project.</p>
        <pre>npm install react</pre>
        <p>More fluff.</p>
        <code>docker build -t app .</code>
        <div class="footer">Copyright</div>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        extracted = extract_relevant_sections(soup)
        
        print(f"Extracted content:\n---\n{extracted}\n---")
        
        self.assertIn("npm install react", extracted)
        self.assertIn("docker build -t app .", extracted)
        self.assertNotIn("Menu", extracted)
        self.assertNotIn("Copyright", extracted)
        print("✓ Web search optimization verified!")

if __name__ == '__main__':
    unittest.main()
