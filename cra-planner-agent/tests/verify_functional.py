
import sys
import unittest
from unittest.mock import MagicMock, patch
import json

# Add src to path
sys.path.append('src')

from parallel_empirical_test import LLMFunctionalVerifier
from agent.validation import DockerBuildTester

class TestFunctionalVerification(unittest.TestCase):

    @patch('parallel_empirical_test.AzureChatOpenAI')
    def test_command_generation(self, mock_azure_openai):
        print("\nTesting Verification Command Generation...")
        
        # Mock LLM response
        mock_llm = mock_azure_openai.return_value
        mock_llm.invoke.return_value.content = "python app.py --version"
        
        verifier = LLMFunctionalVerifier()
        cmd = verifier.generate_verification_command("FROM python:3.9\nCMD python app.py", "my-repo")
        
        self.assertEqual(cmd, "python app.py --version")
        print("✓ Command generation verified!")

    @patch('parallel_empirical_test.AzureChatOpenAI')
    @patch('subprocess.run')
    def test_run_and_verify(self, mock_subprocess, mock_azure_openai):
        print("\nTesting Container Run & Verification Analysis...")
        
        # 1. Mock Container Run (DockerBuildTester)
        tester = DockerBuildTester()
        
        # Mock 'docker image inspect' (success)
        mock_inspect = MagicMock()
        mock_inspect.returncode = 0
        
        # Mock 'docker run' (success output)
        mock_run_res = MagicMock()
        mock_run_res.returncode = 0
        mock_run_res.stdout = "MyApp v1.0.0"
        mock_run_res.stderr = ""
        
        mock_subprocess.side_effect = [mock_inspect, mock_run_res]
        
        run_result = tester.run_container("test-image", "python app.py --version")
        
        self.assertTrue(run_result['success'])
        self.assertIn("MyApp v1.0.0", run_result['output'])
        print("✓ Container execution verified!")

        # 2. Mock LLM Analysis (LLMFunctionalVerifier)
        mock_llm = mock_azure_openai.return_value
        mock_llm.invoke.return_value.content = '```json\n{"success": true, "reason": "Version printed"}\n```'
        
        verifier = LLMFunctionalVerifier()
        analysis = verifier.verify_output("python app.py --version", run_result['output'], 0)
        
        self.assertTrue(analysis['success'])
        self.assertEqual(analysis['reason'], "Version printed")
        print("✓ Output analysis verified!")

if __name__ == '__main__':
    unittest.main()
