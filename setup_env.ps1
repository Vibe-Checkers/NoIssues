# Azure OpenAI Environment Setup Script
# Run this script to set environment variables for the current PowerShell session

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Azure OpenAI Environment Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Get user input for Azure credentials
Write-Host "Please enter your Azure OpenAI credentials:" -ForegroundColor Yellow

# API Key
$env:AZURE_OPENAI_API_KEY = Read-Host "Enter AZURE_OPENAI_API_KEY"

# Endpoint
$env:AZURE_OPENAI_ENDPOINT = Read-Host "Enter AZURE_OPENAI_ENDPOINT (e.g., https://your-resource.openai.azure.com/)"

# Deployment
$env:AZURE_OPENAI_DEPLOYMENT = Read-Host "Enter AZURE_OPENAI_DEPLOYMENT name"

Write-Host ""
Write-Host "Environment variables set!" -ForegroundColor Green
Write-Host ""
Write-Host "Verifying..." -ForegroundColor Yellow
Write-Host "AZURE_OPENAI_API_KEY: $($env:AZURE_OPENAI_API_KEY.Substring(0, [Math]::Min(10, $env:AZURE_OPENAI_API_KEY.Length)))..." -ForegroundColor Gray
Write-Host "AZURE_OPENAI_ENDPOINT: $env:AZURE_OPENAI_ENDPOINT" -ForegroundColor Gray
Write-Host "AZURE_OPENAI_DEPLOYMENT: $env:AZURE_OPENAI_DEPLOYMENT" -ForegroundColor Gray
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run: python example_usage.py" -ForegroundColor White
Write-Host "2. Or run: python test_planner.py" -ForegroundColor White
Write-Host ""
Write-Host "Note: These variables are only set for this session." -ForegroundColor Yellow
Write-Host "To make them permanent, use:" -ForegroundColor Yellow
Write-Host '[Environment]::SetEnvironmentVariable("AZURE_OPENAI_API_KEY", "your-key", "User")' -ForegroundColor Gray

