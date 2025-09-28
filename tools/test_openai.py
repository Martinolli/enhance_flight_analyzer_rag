#!/usr/bin/env python3
"""
Test OpenAI API Key integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_openai_integration():
    """Test if OpenAI API key works"""
    
    print("=== OpenAI API Key Test ===\n")
    
    # Check if API key is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OpenAI API key found in environment")
        return False
    
    print(f"✅ API Key loaded: {api_key[:10]}...{api_key[-10:]}")
    
    # Test OpenAI client
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Simple test call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, RAG test successful!' in exactly those words."}
            ],
            max_tokens=20
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ OpenAI Response: {result}")
        
        if "Hello, RAG test successful!" in result:
            print("🎉 OpenAI integration working perfectly!")
            return True
        else:
            print("⚠️ OpenAI responded but with unexpected content")
            return True
            
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return False

if __name__ == "__main__":
    try:
        success = test_openai_integration()
        if success:
            print("\n✅ Your RAG system is ready to use with OpenAI!")
        else:
            print("\n❌ Please check your OpenAI API key configuration")
    except Exception as e:
        print(f"❌ Test failed: {e}")