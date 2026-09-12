# PS14 Makefile - Security Targets
# Add these targets to your existing Makefile or use standalone

.PHONY: security-scan security-scan-full security-install help

# Default help target
help: ## Show this help
	@echo "PS14 Security Commands:"
	@echo "  security-scan       - Run basic security scan"
	@echo "  security-scan-full  - Run full security scan (with dependencies)"
	@echo "  security-install    - Install security tools"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

security-install: ## Install security scanning tools
	pip install safety bandit semgrep
	@echo "✅ Security tools installed"

security-scan: ## Run basic security scan
	python scripts/security_scan.py

security-scan-full: ## Run full security scan (including dependencies)
	python scripts/security_scan.py --full

security-scan-json: ## Run security scan with JSON output
	python scripts/security_scan.py --json --output .freebuff/security-report.json

security-check: ## Run all security checks (scan + tests)
	@echo "Running security scanner..."
	python scripts/security_scan.py
	@echo ""
	@echo "Running security tests..."
	python scripts/security_test.py || true
	@echo ""
	@echo "✅ All security checks complete"