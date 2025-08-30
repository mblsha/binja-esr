#!/bin/bash

# GitHub Branch Protection Setup Script
# Sets up minimal branch protection with required checks discovery

set -euo pipefail

# Configuration
BRANCH="${1:-master}"               # Branch to protect (default: master)
INCLUDE_ADMINS=true                 # Include admins in protection rules
REQUIRED_APPROVALS=0                # No approvals needed for solo workflow
REQUIRE_UP_TO_DATE=false           # Don't require PR branch to be up-to-date

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prereqs() {
    log "Checking prerequisites..."
    
    # Check if gh is installed
    if ! command -v gh &> /dev/null; then
        error "GitHub CLI (gh) is not installed. Install with: brew install gh"
    fi
    
    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        error "jq is not installed. Install with: brew install jq"
    fi
    
    # Check if authenticated with gh
    if ! gh auth status &> /dev/null; then
        error "Not authenticated with GitHub CLI. Run: gh auth login"
    fi
    
    # Check if in a git repo
    if ! git rev-parse --git-dir &> /dev/null; then
        error "Not in a git repository"
    fi
    
    success "Prerequisites check passed"
}

# Get repository information
get_repo_info() {
    log "Getting repository information..."
    
    REPO_SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
    if [[ -z "$REPO_SLUG" ]]; then
        error "Could not determine repository information"
    fi
    
    log "Target repo: ${REPO_SLUG} | Branch to protect: ${BRANCH}"
}

# Check if branch exists remotely
check_branch_exists() {
    log "Checking if branch '$BRANCH' exists on remote..."
    
    if ! git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null; then
        error "Remote branch '$BRANCH' not found on origin"
    fi
    
    success "Branch '$BRANCH' exists on remote"
}

# Create probe PR to discover checks
create_probe_pr() {
    log "Creating probe PR to discover GitHub Actions checks..."
    
    TMP_BRANCH="bot/protection-probe-$(date +%s)"
    
    # Create temporary branch
    git switch -c "$TMP_BRANCH"
    git commit --allow-empty -m "protection probe: trigger checks"
    git push -u origin "$TMP_BRANCH"
    
    # Create PR
    gh pr create -H "$TMP_BRANCH" -B "$BRANCH" -t "Protection Probe" -b "Temporary PR to discover check runs" --draft
    
    PR_NUMBER="$(gh pr view --json number -q .number)"
    log "Created probe PR #$PR_NUMBER"
}

# Wait for and collect check runs
collect_checks() {
    log "Waiting for checks to start on PR #$PR_NUMBER..."
    
    PR_HEAD_SHA="$(gh pr view --json headRefOid -q .headRefOid)"
    
    # Wait for checks to start (up to 3 minutes)
    for i in {1..60}; do
        COUNT="$(gh api repos/:owner/:repo/commits/$PR_HEAD_SHA/check-runs -q '.total_count')"
        if [[ "$COUNT" -gt 0 ]]; then
            break
        fi
        echo -n "."
        sleep 3
    done
    echo ""
    
    if [[ "$COUNT" -eq 0 ]]; then
        warn "No checks found. This might be because:"
        warn "  1. No GitHub Actions workflows are configured for pull_request events"
        warn "  2. Checks haven't started yet (workflow may be slow)"
        warn "  3. Repository has no CI/CD setup"
        warn ""
        warn "Continuing with empty checks list..."
        REQ_CHECKS="[]"
    else
        # Build the 'checks' array for branch protection API
        REQ_CHECKS="$(
            gh api repos/:owner/:repo/commits/$PR_HEAD_SHA/check-runs \
                -q '[.check_runs[] | {context: .name, app_id: .app.id}]
                    | sort_by(.context, .app_id)
                    | unique'
        )"
        
        log "Discovered checks:"
        echo "$REQ_CHECKS" | jq .
    fi
}

# Apply branch protection
apply_protection() {
    log "Applying branch protection to '$BRANCH'..."
    
    # Create protection configuration
    if [[ "$REQ_CHECKS" == "[]" ]]; then
        # No checks found - use null for required_status_checks
        cat > /tmp/protection.json <<JSON
{
  "required_status_checks": null,
  "enforce_admins": ${INCLUDE_ADMINS},
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false,
    "required_approving_review_count": ${REQUIRED_APPROVALS},
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": false,
  "lock_branch": false
}
JSON
    else
        # Checks found - use them in required_status_checks
        cat > /tmp/protection.json <<JSON
{
  "required_status_checks": {
    "strict": ${REQUIRE_UP_TO_DATE},
    "checks": ${REQ_CHECKS}
  },
  "enforce_admins": ${INCLUDE_ADMINS},
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false,
    "required_approving_review_count": ${REQUIRED_APPROVALS},
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": false,
  "lock_branch": false
}
JSON
    fi

    # Apply protection
    if gh api \
        --method PUT \
        -H "Accept: application/vnd.github+json" \
        "/repos/:owner/:repo/branches/${BRANCH}/protection" \
        --input /tmp/protection.json > /dev/null; then
        success "Branch protection applied successfully!"
    else
        error "Failed to apply branch protection"
    fi
    
    # Clean up temp file
    rm /tmp/protection.json
}

# Clean up probe PR and branch
cleanup_probe() {
    log "Cleaning up probe PR and branch..."
    
    gh pr close "$PR_NUMBER" --delete-branch
    git switch "$BRANCH" 2>/dev/null || git switch -c "$BRANCH" origin/"$BRANCH"
    
    success "Cleanup completed"
}

# Test protection
test_protection() {
    log "Testing branch protection..."
    
    # Test direct push (should fail)
    log "Testing direct push to protected branch (should fail)..."
    if git push origin HEAD:"$BRANCH" 2>/dev/null; then
        warn "Direct push succeeded - protection may not be working correctly"
    else
        success "Direct push blocked as expected"
    fi
}

# Main execution
main() {
    echo "GitHub Branch Protection Setup"
    echo "=============================="
    echo ""
    
    check_prereqs
    get_repo_info
    check_branch_exists
    
    # Ask for confirmation
    echo ""
    echo "This will:"
    echo "  - Create a temporary probe PR to discover GitHub Actions checks"
    echo "  - Apply branch protection to '$BRANCH' requiring:"
    echo "    * All changes go through PRs"
    echo "    * All discovered checks must pass"
    echo "    * No force pushes allowed"
    echo "    * Admins are subject to the same rules"
    echo "    * No approval requirements (solo-dev friendly)"
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Aborted by user"
        exit 0
    fi
    
    create_probe_pr
    collect_checks
    apply_protection
    cleanup_probe
    test_protection
    
    echo ""
    success "Branch protection setup completed for ${REPO_SLUG}@${BRANCH}!"
    echo ""
    echo "Next steps:"
    echo "  1. All future changes must go through PRs"
    echo "  2. Create feature branches with: git switch -c feature/my-feature"
    echo "  3. Push and create PRs with: gh pr create"
    echo "  4. Merge when checks pass with: gh pr merge"
    echo ""
}

# Show help
show_help() {
    echo "Usage: $0 [BRANCH]"
    echo ""
    echo "Sets up GitHub branch protection for the specified branch (default: master)"
    echo ""
    echo "Examples:"
    echo "  $0              # Protect 'master' branch"
    echo "  $0 main         # Protect 'main' branch"
    echo ""
    echo "Prerequisites:"
    echo "  - GitHub CLI (gh) installed and authenticated"
    echo "  - jq installed"
    echo "  - Repository admin permissions"
    echo "  - Clean working tree"
}

# Handle help flag
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"