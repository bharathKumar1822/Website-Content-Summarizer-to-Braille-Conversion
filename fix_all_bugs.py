"""
Comprehensive bug fix script for the Website-Content-Summarizer-to-Braille-Conversion project.
Run this script once to apply all fixes.
"""

import re
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def fix_enhanced_summarization():
    """Fix bugs in enhanced_summarization.py"""
    filepath = os.path.join(PROJECT_DIR, 'enhanced_summarization.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    changes = []

    # Fix 1: Add _MODEL_CACHE global definition (referenced but never initialized)
    if '_MODEL_CACHE = {}' not in content:
        old = 'logger = logging.getLogger(\'enhanced_summarization\')'
        new = (
            'logger = logging.getLogger(\'enhanced_summarization\')\n\n'
            '# In-memory model cache for reuse across requests\n'
            '_MODEL_CACHE = {}'
        )
        if old in content:
            content = content.replace(old, new, 1)
            changes.append('Added _MODEL_CACHE global dict definition')

    # Fix 2: Fix summarize_with_gpt to work with openai v0.28.x (installed version)
    # The function tries `from openai import OpenAI` which only exists in v1.x+
    # We need to support both old and new APIs gracefully.
    old_func_signature = "def summarize_with_gpt(text, max_tokens=800):"
    if old_func_signature in content:
        # Find the full function and replace it
        func_start = content.find(old_func_signature)
        # Find the next function definition after this one
        next_func = content.find('\ndef ', func_start + len(old_func_signature))
        old_func = content[func_start:next_func]

        new_func = '''def summarize_with_gpt(text, max_tokens=800):
    """
    Summarize text using OpenAI's GPT-3.5-turbo model with a direct approach.
    Updated to generate longer summaries (12-15 lines).
    Compatible with both openai v0.x (legacy/0.28.x) and v1.x (new) APIs.
    """
    try:
        # Get API key from environment
        api_key = os.environ.get('OPENAI_API_KEY')

        if not api_key:
            logger.warning("No OpenAI API key found in environment")
            return None

        logger.info(f"Using API key: {api_key[:10]}...")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text accurately and comprehensively."
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following text in a detailed manner, providing a comprehensive "
                    "summary of 12-15 lines that captures all important information and key points:"
                    f"\\n\\n{text}"
                )
            }
        ]

        # Try new openai v1.x API first (OpenAI class)
        try:
            from openai import OpenAI as _OpenAI
            client = _OpenAI(api_key=api_key)
            logger.info("Using OpenAI v1.x client")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens
            )
            summary = response.choices[0].message.content.strip()
            logger.info(f"OpenAI summary length: {len(summary)} characters")
            return summary
        except (ImportError, AttributeError):
            pass  # Fall through to legacy v0.x API

        # Fall back to old openai v0.x API (openai==0.28.x)
        import openai as _openai
        logger.info("Using OpenAI v0.x API (legacy)")
        _openai.api_key = api_key
        response = _openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens
        )
        summary = response.choices[0].message['content'].strip()
        logger.info(f"OpenAI summary length: {len(summary)} characters")
        return summary

    except Exception as e:
        logger.error(f"Error in OpenAI summarization: {e}")
        import traceback
        traceback.print_exc()
        return None

'''
        content = content[:func_start] + new_func + content[next_func:]
        changes.append('Fixed summarize_with_gpt to work with openai v0.28.x and v1.x')

    # Fix 3: Fix TRANSFORMERS_OFFLINE=1 blocking GPT-2 model loading
    # The env var is set in app.py, which blocks HuggingFace downloads even for GPT-2.
    # In summarize_with_gpt2, temporarily unset it before loading model.
    old_gpt2_load = "        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')"
    new_gpt2_load = (
        "        # Temporarily allow network access for model download if not already cached\n"
        "        _orig_offline = os.environ.pop('TRANSFORMERS_OFFLINE', None)\n"
        "        try:\n"
        "            tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=CACHE_DIR)\n"
        "        finally:\n"
        "            if _orig_offline is not None:\n"
        "                os.environ['TRANSFORMERS_OFFLINE'] = _orig_offline"
    )
    if old_gpt2_load in content and "Temporarily allow network access" not in content:
        content = content.replace(old_gpt2_load, new_gpt2_load, 1)
        changes.append('Fixed GPT-2 loader to bypass TRANSFORMERS_OFFLINE for model download')

    old_gpt2_model_load = "        model = GPT2LMHeadModel.from_pretrained('gpt2')"
    new_gpt2_model_load = "            model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=CACHE_DIR)"
    if old_gpt2_model_load in content:
        content = content.replace(old_gpt2_model_load, new_gpt2_model_load, 1)
        changes.append('Fixed GPT-2 model load to use cache_dir')

    # Fix 4: Gemini API — handle 429 (quota exceeded) with exponential backoff / try flash-lite fallback
    # The summarize_with_gemini function already has fallback to gemini-pro.
    # Let's add a check: if status_code == 429 (quota exceeded), do NOT continue retrying
    # but instead surface a clear log message and immediately return None to trigger fallback.
    old_quota_check = (
        '        else:\n'
        '            # If Gemini Pro 1.5 fails, try with Gemini Pro as fallback\n'
        '            if response.status_code == 400 and "gemini-2.0-flash" in response.text:\n'
        '                logger.warning("Gemini Pro 1.5 not available, falling back to Gemini Pro")\n'
        '                return summarize_with_gemini_pro(text, min_length, max_length)\n'
        '            \n'
        '            logger.error(f"Gemini API request failed with status code {response.status_code}: {response.text}")\n'
        '            return None'
    )
    new_quota_check = (
        '        else:\n'
        '            # Handle quota exceeded (429) - return None to trigger fallback immediately\n'
        '            if response.status_code == 429:\n'
        '                logger.warning("Gemini API quota exceeded (429), switching to fallback summarizer")\n'
        '                return None\n'
        '            # If model not found (400), try with Gemini Flash as fallback\n'
        '            if response.status_code == 400:\n'
        '                logger.warning("Gemini model not available (400), falling back to gemini-pro")\n'
        '                return summarize_with_gemini_pro(text, min_length, max_length)\n'
        '            logger.error(f"Gemini API request failed with status code {response.status_code}: {response.text}")\n'
        '            return None'
    )
    if old_quota_check in content:
        content = content.replace(old_quota_check, new_quota_check, 1)
        changes.append('Fixed Gemini 429 quota handling to immediately return None instead of retrying')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n[enhanced_summarization.py] Applied {len(changes)} fixes:")
    for c in changes:
        print(f"  - {c}")
    return changes


def fix_app_py():
    """Fix bugs in app.py"""
    filepath = os.path.join(PROJECT_DIR, 'app.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    changes = []

    # Fix 1: Cache retrieval bug - pickle.loads called multiple times on same bytes
    # Lines ~638-657: deserializes cached_result 5+ times; each time gets a new dict object
    # so mutation (like setting ['cache_hit']) is discarded.
    old_cache_block = '''            try:
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for URL: {url}, language: {target_language}")
                    
                    # Always regenerate audio data even for cached results
                    # This ensures fresh audio for each request
                    if target_language in ['en', 'hi', 'kn']:
                        text_for_audio = pickle.loads(cached_result).get('translated_summary') if target_language != 'en' and pickle.loads(cached_result).get('translated_summary') else pickle.loads(cached_result).get('summary')
                        if text_for_audio:
                            pickle.loads(cached_result)['audio'] = generate_audio_google_tts(text_for_audio, target_language)
                    
                    # Add cache hit indicator and processing time 
                    pickle.loads(cached_result)['cache_hit'] = True
                    pickle.loads(cached_result)['processing_time'] = 0  # No processing time for cache hits
                    return jsonify(pickle.loads(cached_result))
            except Exception as cache_error:
                logger.error(f"Error retrieving from cache: {str(cache_error)}", exc_info=True)
                # Continue with normal processing if cache retrieval fails'''

    new_cache_block = '''            try:
                cached_bytes = redis_client.get(cache_key)
                if cached_bytes:
                    logger.info(f"Cache hit for URL: {url}, language: {target_language}")
                    # Deserialize ONCE to avoid creating multiple separate objects
                    cached_result = pickle.loads(cached_bytes)

                    # Always regenerate audio data even for cached results
                    # This ensures fresh audio for each request
                    if target_language in ['en', 'hi', 'kn']:
                        text_for_audio = (
                            cached_result.get('translated_summary')
                            if target_language != 'en' and cached_result.get('translated_summary')
                            else cached_result.get('summary')
                        )
                        if text_for_audio:
                            cached_result['audio'] = generate_audio_google_tts(text_for_audio, target_language)

                    # Add cache hit indicator and processing time
                    cached_result['cache_hit'] = True
                    cached_result['processing_time'] = 0  # No processing time for cache hits
                    return jsonify(cached_result)
            except Exception as cache_error:
                logger.error(f"Error retrieving from cache: {str(cache_error)}", exc_info=True)
                # Continue with normal processing if cache retrieval fails'''

    if old_cache_block in content:
        content = content.replace(old_cache_block, new_cache_block, 1)
        changes.append('Fixed cache retrieval bug (pickle.loads called 5x on same bytes object)')
    else:
        print("  WARN: Could not find exact cache block to fix - may already be patched")

    # Fix 2: /correct-voice-command endpoint - cache_key undefined when OPENAI_API_KEY not set
    # The variable `cache_key` is defined inside `if OPENAI_API_KEY:` but referenced later
    # outside that block when saving to cache.
    old_cache_key_init = '''    # Initialize variables
    corrected_command = original_command
    correction_method = 'none'
    correction_confidence = confidence'''

    new_cache_key_init = '''    # Initialize variables
    corrected_command = original_command
    correction_method = 'none'
    correction_confidence = confidence
    # Pre-compute cache key so it's available throughout the function
    cache_key = generate_cache_key('command_correction', {
        'command': original_command,
        'confidence': confidence
    })'''

    if old_cache_key_init in content:
        content = content.replace(old_cache_key_init, new_cache_key_init, 1)
        changes.append('Fixed cache_key undefined variable in /correct-voice-command')

    # Fix 3: /correct-voice-command - Remove duplicate cache_key definition inside if block
    old_duplicate_cache_key = '''    # Try GPT-4 correction if API key is available
    if OPENAI_API_KEY:
        try:
            # Cache key for this correction request
            cache_key = generate_cache_key('command_correction', {
                'command': original_command,
                'confidence': confidence
            })
            
            # Check cache first'''

    new_no_duplicate_cache_key = '''    # Try GPT-4 correction if API key is available
    if OPENAI_API_KEY:
        try:
            # Check cache first (cache_key already defined above)'''

    if old_duplicate_cache_key in content:
        content = content.replace(old_duplicate_cache_key, new_no_duplicate_cache_key, 1)
        changes.append('Removed duplicate cache_key definition in /correct-voice-command')

    # Fix 4: /correct-voice-command - gpt-4 model not accessible, use gpt-3.5-turbo
    # Also update to use API compatible with openai v0.28.x
    old_gpt4_call = '''            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Use GPT-4 for best accuracy
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=50,
                temperature=0.3  # Low temperature for more predictable outputs
            )
            
            # Extract the corrected command from the response
            gpt_corrected_command = response.choices[0].message.content.strip()'''

    new_gpt35_call = '''            # Call OpenAI API (using gpt-3.5-turbo for broader availability)
            # Try new v1.x API first
            try:
                from openai import OpenAI as _OpenAI
                _client = _OpenAI(api_key=OPENAI_API_KEY)
                _resp = _client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=50,
                    temperature=0.3
                )
                gpt_corrected_command = _resp.choices[0].message.content.strip()
            except (ImportError, AttributeError):
                # Fall back to openai v0.28.x API
                _resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=50,
                    temperature=0.3
                )
                gpt_corrected_command = _resp.choices[0].message['content'].strip()'''

    if old_gpt4_call in content:
        content = content.replace(old_gpt4_call, new_gpt35_call, 1)
        changes.append('Replaced inaccessible gpt-4 with gpt-3.5-turbo in /correct-voice-command')

    # Fix 5: Redis connection resilience - wrap redis calls to auto-detect connection drops
    # The global REDIS_AVAILABLE stays True even when connection drops.
    # Add a helper that re-checks before each operation.
    old_check_redis = '''def check_redis_connection():
    """Check if Redis connection is working and return status."""
    if not REDIS_AVAILABLE:
        return False
    
    try:
        redis_client.ping()
        return True
    except Exception:
        return False'''

    new_check_redis = '''def check_redis_connection():
    """Check if Redis connection is working and return status."""
    global REDIS_AVAILABLE
    if not redis_client:
        return False

    try:
        redis_client.ping()
        REDIS_AVAILABLE = True
        return True
    except Exception:
        # Mark as unavailable; will retry on next request
        REDIS_AVAILABLE = False
        return False


def safe_redis_get(key):
    """Safe Redis GET that handles connection drops gracefully."""
    if not check_redis_connection():
        return None
    try:
        return redis_client.get(key)
    except Exception as e:
        logger.warning(f"Redis GET failed: {e}")
        return None


def safe_redis_setex(key, ttl, value):
    """Safe Redis SETEX that handles connection drops gracefully."""
    if not check_redis_connection():
        return False
    try:
        redis_client.setex(key, ttl, value)
        return True
    except Exception as e:
        logger.warning(f"Redis SETEX failed: {e}")
        return False'''

    if old_check_redis in content:
        content = content.replace(old_check_redis, new_check_redis, 1)
        changes.append('Added safe Redis helpers with automatic connection recovery')

    # Fix 6: Use safe_redis_get/setex in process_url cache check
    old_redis_cache_check = "                cached_bytes = redis_client.get(cache_key)"
    new_redis_cache_check = "                cached_bytes = safe_redis_get(cache_key)"
    if old_redis_cache_check in content:
        content = content.replace(old_redis_cache_check, new_redis_cache_check, 1)
        changes.append('Used safe_redis_get in process_url cache retrieval')

    # Fix 7: Use safe_redis_setex in process_url caching
    old_redis_save = '''                    redis_client.setex(
                        cache_key,
                        86400,  # 24 hours in seconds
                        pickle.dumps(cache_result)
                    )'''
    new_redis_save = '''                    safe_redis_setex(
                        cache_key,
                        86400,  # 24 hours in seconds
                        pickle.dumps(cache_result)
                    )'''
    if old_redis_save in content:
        content = content.replace(old_redis_save, new_redis_save, 1)
        changes.append('Used safe_redis_setex in process_url cache save')

    # Fix 8: Use safe_redis_get in translate_text cache check
    old_trans_cache_get = "            cached_result = redis_client.get(cache_key)"
    new_trans_cache_get = "            cached_result = safe_redis_get(cache_key)"
    # This pattern appears in multiple places - replace translate cache specifically
    old_trans_block = '''    try:
        # Check cache first
        if REDIS_AVAILABLE:
            cache_key = generate_cache_key('translate', {'text': text, 'lang': target_language})
            cached_result = redis_client.get(cache_key)'''
    new_trans_block = '''    try:
        # Check cache first
        if check_redis_connection():
            cache_key = generate_cache_key('translate', {'text': text, 'lang': target_language})
            cached_result = safe_redis_get(cache_key)'''
    if old_trans_block in content:
        content = content.replace(old_trans_block, new_trans_block, 1)
        changes.append('Used safe_redis_get in translate_text cache check')

    # Fix 9: Use safe_redis_setex in translate_text cache save
    old_trans_save = '''        if REDIS_AVAILABLE and translated_text:
            # Cache translations for 7 days (604800 seconds)
            redis_client.setex(cache_key, 604800, pickle.dumps(translated_text))'''
    new_trans_save = '''        if translated_text:
            # Cache translations for 7 days (604800 seconds)
            safe_redis_setex(cache_key, 604800, pickle.dumps(translated_text))'''
    if old_trans_save in content:
        content = content.replace(old_trans_save, new_trans_save, 1)
        changes.append('Used safe_redis_setex in translate_text cache save')

    # Fix 10: Use safe helpers in audio TTS cache
    old_audio_cache_get = '''        # Check cache first
        if REDIS_AVAILABLE:
            cache_key = generate_cache_key('audio', {'text': text, 'lang': language})
            cached_result = redis_client.get(cache_key)'''
    new_audio_cache_get = '''        # Check cache first
        if check_redis_connection():
            cache_key = generate_cache_key('audio', {'text': text, 'lang': language})
            cached_result = safe_redis_get(cache_key)'''
    if old_audio_cache_get in content:
        content = content.replace(old_audio_cache_get, new_audio_cache_get, 1)
        changes.append('Used safe_redis_get in audio TTS cache retrieval')

    old_audio_cache_save = '''        if REDIS_AVAILABLE and audio_data:
            # Cache audio for 7 days (604800 seconds)
            redis_client.setex(cache_key, 604800, pickle.dumps(audio_data))'''
    new_audio_cache_save = '''        if audio_data:
            # Cache audio for 7 days (604800 seconds)
            safe_redis_setex(cache_key, 604800, pickle.dumps(audio_data))'''
    if old_audio_cache_save in content:
        content = content.replace(old_audio_cache_save, new_audio_cache_save, 1)
        changes.append('Used safe_redis_setex in audio TTS cache save')

    # Fix 11: Undefined title variable - add default
    # When newspaper3k succeeds but BeautifulSoup title fetch fails, 'title' might be undefined
    old_title_fallback = '''            else:
                # If using newspaper3k, we need to get the title separately
                try:
                    response = requests.get(url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }, verify=False)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.title.string if soup.title else "No title found"
                except:
                    title = "No title found"'''

    new_title_fallback = '''            else:
                # If using newspaper3k, we need to get the title separately
                title = "No title found"  # Default in case fetch fails
                try:
                    response = requests.get(url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }, verify=False, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.title.string if soup.title else "No title found"
                except Exception:
                    title = "No title found"'''

    if old_title_fallback in content:
        content = content.replace(old_title_fallback, new_title_fallback, 1)
        changes.append('Fixed potential undefined title variable; added timeout to title fetch')

    # Fix 12: Add timeout to the main URL extraction request
    old_url_fetch = '''                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }, verify=False)
                response.raise_for_status()
                
                # Parse HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                article_content = extract_main_content(soup)
                title = soup.title.string if soup.title else "No title found"'''

    new_url_fetch = '''                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }, verify=False, timeout=20)
                response.raise_for_status()

                # Parse HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                article_content = extract_main_content(soup)
                title = soup.title.string if soup.title else "No title found"'''

    if old_url_fetch in content:
        content = content.replace(old_url_fetch, new_url_fetch, 1)
        changes.append('Added timeout=20 to fallback URL fetch request')

    # Fix 13: Add voice command cache check in /correct-voice-command
    old_vc_cache_check = '''            # Check cache first
            if REDIS_AVAILABLE:
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    return pickle.loads(cached_result)'''

    new_vc_cache_check = '''            # Check cache first
            if check_redis_connection():
                cached_bytes = safe_redis_get(cache_key)
                if cached_bytes:
                    return jsonify(pickle.loads(cached_bytes))'''

    if old_vc_cache_check in content:
        content = content.replace(old_vc_cache_check, new_vc_cache_check, 1)
        changes.append('Fixed voice command cache: returns jsonify response, uses safe_redis_get')

    # Fix 14: Use safe_redis_setex in voice command cache save
    old_vc_cache_save = '''    if REDIS_AVAILABLE:
        try:
            redis_client.setex(
                cache_key,
                60 * 60,  # Cache for 1 hour
                pickle.dumps(result)
            )
        except Exception as e:
            logger.error(f"Error caching command correction: {e}")'''

    new_vc_cache_save = '''    safe_redis_setex(cache_key, 60 * 60, pickle.dumps(result))  # Cache for 1 hour'''

    if old_vc_cache_save in content:
        content = content.replace(old_vc_cache_save, new_vc_cache_save, 1)
        changes.append('Used safe_redis_setex in voice command correction cache save')

    # Fix 15: Add logging for TRANSFORMERS_OFFLINE suppression of GPT-2
    old_transformers_offline = "os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Avoid downloading models during runtime"
    new_transformers_offline = (
        "# Note: TRANSFORMERS_OFFLINE=1 prevents HuggingFace downloads at runtime.\n"
        "# GPT-2 in enhanced_summarization.py bypasses this for initial model download.\n"
        "os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Avoid downloading models during runtime"
    )
    if old_transformers_offline in content:
        content = content.replace(old_transformers_offline, new_transformers_offline, 1)
        changes.append('Added clarifying comment for TRANSFORMERS_OFFLINE setting')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n[app.py] Applied {len(changes)} fixes:")
    for c in changes:
        print(f"  - {c}")
    return changes


def verify_fixes():
    """Run basic syntax check on modified files."""
    import subprocess
    import sys

    files = ['app.py', 'enhanced_summarization.py']
    all_ok = True
    print("\n[Verification] Syntax checking modified files...")
    for fname in files:
        fpath = os.path.join(PROJECT_DIR, fname)
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', fpath],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✓ {fname} - syntax OK")
        else:
            print(f"  ✗ {fname} - SYNTAX ERROR:")
            print(result.stderr)
            all_ok = False
    return all_ok


if __name__ == '__main__':
    print("=" * 60)
    print("Bug Fix Script - Website Content Summarizer to Braille")
    print("=" * 60)

    fix_enhanced_summarization()
    fix_app_py()
    ok = verify_fixes()

    print("\n" + "=" * 60)
    if ok:
        print("All fixes applied successfully. Files pass syntax check.")
    else:
        print("Some fixes applied but syntax errors detected. Review above.")
    print("=" * 60)
