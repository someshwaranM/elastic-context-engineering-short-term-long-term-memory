# LangGraph Agent Memory v3 - New Features

## Overview

Version 3 adds critical features for **context isolation**, **conflict resolution**, and **multi-user data isolation** to prevent context poisoning and context clash issues.

## Key Problems Solved

### 1. Context Poisoning Prevention
**Problem**: Malicious or incorrect information can contaminate long-term memory, causing persistent errors.

**Solution**: 
- **Context Validation**: All content is validated before indexing using LLM-based validation
- **Context Quarantine**: Suspicious or invalid content is isolated in separate threads
- **Validation Scoring**: Each piece of content receives a validation score (0.0-1.0)

### 2. Conflict Resolution
**Problem**: When users correct information (e.g., "I work at Google" → "Actually, I work at Microsoft"), both facts exist in memory, causing confusion.

**Solution**:
- **Conflict Detection**: Automatically detects contradictory facts for the same user
- **Fact Tracking**: Tracks fact keys (e.g., "workplace", "name", "location")
- **Automatic Resolution**: Prefers newer information and quarantines old conflicting data
- **Correction Tracking**: Marks corrections and links them to the original messages

### 3. Multi-User Data Isolation
**Problem**: Multiple users accessing the same system could see each other's data, causing privacy issues and confusion.

**Solution**:
- **User ID Isolation**: All data is filtered by `user_id` at retrieval time
- **Thread Isolation**: Each user gets their own thread ID
- **Quarantine Isolation**: Quarantined content is isolated per user

## New Features

### 1. Context Validation (`validate_context()`)
- Validates content before indexing
- Checks for:
  - Factual accuracy
  - Safety and appropriateness
  - Legitimacy (not spam/poisoning)
  - Coherence
- Returns validation score and risk level
- Invalid content is quarantined automatically

### 2. Conflict Detection (`detect_conflicts()`)
- Extracts key facts from new content
- Searches for existing conflicting facts for the same user
- Detects contradictions (e.g., "Google" vs "Microsoft")
- Recommends resolution strategy:
  - `prefer_newer`: Use most recent information
  - `prefer_new`: Use new information
  - `quarantine`: Isolate conflicting content
  - `merge`: Attempt to merge information

### 3. Context Quarantine (`quarantine_context()`)
- Isolates suspicious or conflicting content
- Prevents bad information from spreading
- Creates separate quarantine threads
- Marks documents with `is_quarantined: true`
- Quarantined content is excluded from normal retrieval

### 4. Multi-User Isolation
- **User ID Required**: Each session requires a user ID
- **Filtered Retrieval**: All searches automatically filter by `user_id`
- **Quarantine Isolation**: Quarantined content is user-specific
- **Thread Isolation**: Each user has their own conversation threads

## Elasticsearch Schema Updates

New fields added to the index mapping:

```json
{
  "user_id": "keyword",              // User ID for isolation
  "is_quarantined": "boolean",       // Whether content is quarantined
  "quarantine_reason": "text",       // Reason for quarantine
  "has_conflicts": "boolean",         // Whether conflicts detected
  "conflict_info": "object",          // Conflict details
  "validated": "boolean",             // Whether validation passed
  "validation_score": "float",       // Validation confidence (0.0-1.0)
  "is_correction": "boolean",        // Whether this is a correction
  "corrects_message_id": "keyword", // ID of message being corrected
  "fact_key": "keyword"              // Fact category (e.g., "workplace")
}
```

## Usage Examples

### Example 1: Conflict Resolution

```
User: "My name is Alice and I work at Google"
→ Stored to Elasticsearch ✅
→ fact_key: "workplace", fact_value: "Google"

User: "Actually, I work at Microsoft" (correction)
→ Conflict detected: workplace "Google" vs "Microsoft"
→ Resolution: prefer_newer
→ Old "Google" fact quarantined
→ New "Microsoft" fact stored ✅
→ is_correction: true

User: "Where do I work?"
→ Retrieves ONLY "Microsoft" (old "Google" is quarantined)
→ Model says "You work at Microsoft" ✅
```

### Example 2: Multi-User Isolation

```
User 1 (user_id: "alice"):
  "My name is Alice and I work at Google"
  → Stored with user_id: "alice"

User 2 (user_id: "bob"):
  "My name is Bob and I work at Microsoft"
  → Stored with user_id: "bob"

User 1 asks: "Where do I work?"
→ Retrieves ONLY user_id: "alice" documents
→ Returns "Google" (Bob's data is isolated) ✅
```

### Example 3: Context Validation

```
User: "The sky is green and 2+2=5"
→ Validation: is_valid: false, risk_level: "high"
→ Reason: "Contains obviously false information"
→ Content quarantined, NOT indexed ✅
```

## Command Line Arguments

New arguments in v3:

```bash
--user-id USER_ID              # User ID for multi-user isolation
--enable-validation            # Enable context validation (default: True)
--enable-conflict-detection    # Enable conflict detection (default: True)
--enable-quarantine           # Enable context quarantine (default: True)
```

## How It Works

### Indexing Flow

1. **Extract Messages** from checkpoints
2. **Validate** each message (if enabled)
   - Invalid → Quarantine, skip indexing
3. **Detect Conflicts** with existing facts (if enabled)
   - Conflicts found → Resolve strategy
   - Old conflicting facts → Quarantine
4. **Extract Fact Keys** for tracking
5. **Index** with metadata (user_id, validation_score, conflict_info, etc.)

### Retrieval Flow

1. **User Query** received
2. **Generate Embedding** for query
3. **Search Elasticsearch** with filters:
   - `user_id` must match current user
   - `is_quarantined` must be false
4. **Prune** context (Provence)
5. **Summarize** context
6. **Check Relevance** (LLM-based)
7. **Return** validated, conflict-free context

## Benefits

1. **Prevents Context Poisoning**: Invalid content is quarantined before indexing
2. **Resolves Conflicts**: Automatically handles corrections and updates
3. **Multi-User Safe**: Complete data isolation between users
4. **Maintains Data Quality**: Only validated, conflict-free information is used
5. **Traceable**: All validation and conflict decisions are logged

## Migration from v2

- **Backward Compatible**: Existing indices will work, but new fields will be added
- **User ID Required**: v3 requires user_id (prompts if not provided)
- **Automatic Filtering**: All retrievals automatically filter by user_id
- **Quarantine Safe**: Quarantined content is excluded from searches

## Important Notes

1. **User ID is Critical**: Without proper user_id, data isolation fails
2. **Validation Can Be Strict**: Some legitimate content might be rejected if validation is too strict
3. **Conflict Resolution**: Currently uses "prefer newer" strategy - can be customized
4. **Quarantine is Permanent**: Quarantined content is not automatically restored

## Future Enhancements

- Configurable conflict resolution strategies
- Quarantine review and restoration
- Fact versioning and history
- Cross-user fact sharing (with permissions)
- Advanced validation rules

