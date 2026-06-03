
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()  

@dataclass
class CoachingTip:
    tip:         str  = ""    
    model_used:  str  = ""
    prompt_used: str  = ""
    fallback:    bool = False  

class LLMCoach:
  

    MODEL   = "llama-3.3-70b-versatile"     
    MAX_TOK = 180                  

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client  = None

        if self._api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key)
            except ImportError:
                print("[llm] groq package not installed — run: pip install groq")
        else:
            print("[llm] GROQ_API_KEY not set — coaching tips will use fallback text")

   

    def generate(self, pipeline_result) -> CoachingTip:
        """
        Generate a coaching tip from a PipelineResult.
        Accepts any object with a to_llm_context() method,
        or a plain dict.
        """
        ctx = (
            pipeline_result.to_llm_context()
            if hasattr(pipeline_result, "to_llm_context")
            else pipeline_result
        )
        prompt = self._build_prompt(ctx)

        if self._client is None:
            return CoachingTip(
                tip=_rule_based_fallback(ctx),
                model_used="fallback",
                prompt_used=prompt,
                fallback=True,
            )

        try:
            response = self._client.chat.completions.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOK,
                temperature=0.65,       # slight creativity, stays grounded
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            tip_text = response.choices[0].message.content.strip()
            return CoachingTip(
                tip=tip_text,
                model_used=self.MODEL,
                prompt_used=prompt,
            )

        except Exception as e:
            print(f"[llm] Groq call failed: {e}")
            return CoachingTip(
                tip=_rule_based_fallback(ctx),
                model_used="fallback",
                prompt_used=prompt,
                fallback=True,
            )

    def generate_from_dict(self, context: dict) -> CoachingTip:
        """Convenience: pass the dict directly."""
        return self.generate(context)

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_prompt(self, ctx: dict) -> str:
        """
        Build a structured prompt. The more specific the numbers,
        the more specific the coaching tip.
        """
        shot            = ctx.get("shot", "unknown shot")
        confidence      = ctx.get("shot_confidence", "unknown")
        overall         = ctx.get("technique_overall", 0)
        best_match = "unknown"
        flags           = ctx.get("pose_flags", [])
        if flags:
            best_match = flags[0].replace("Best Match: ", "")
        
        length_zone     = ctx.get("length_zone", "unknown")
        line_zone       = ctx.get("line_zone", "unknown")
        handedness      = ctx.get("handedness", "right")

        flags_str = (
            "\n  - " + "\n  - ".join(flags)
            if flags else "  None detected"
        )

        return textwrap.dedent(f"""
Shot Classification : {shot} ({confidence})
Pose Match          : {best_match}
Similarity Score    : {overall}/100

Delivery:
  Length Zone       : {length_zone}
  Line Zone         : {line_zone}

Provide:
1. What shot the batsman appears to be attempting
2. One technical strength
3. One technical weakness
4. One actionable coaching suggestion

Keep response under 4 sentences.
""").strip()



_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert cricket batting coach with 20 years of experience coaching
    first-class and international batsmen.

    Rules for your response:
    - Give exactly 2–3 sentences of coaching advice. No more.
    - Be specific to the shot played and the technique flags given.
    - Reference the actual numbers (scores, zones) in your advice.
    - Use technical cricket terms (elbow position, weight transfer,
      follow-through, head position, base stance) where relevant.
    - Do NOT give generic advice like "practice more" or "watch the ball".
    - Do NOT introduce yourself or add a preamble.
    - Start directly with the coaching tip.
""").strip()




def _rule_based_fallback(ctx):

    sim = ctx.get("technique_overall", 0)
    flags = ctx.get("pose_flags", [])
    

    best_match = "unknown"

    if flags:
        best_match = flags[0].replace("Best Match: ", "")

    return (
        f"The batting technique most closely matches "
        f"'{best_match}' with a similarity score of "
        f"{sim:.1f}/100. Focus on improving body alignment "
        f"and consistency through the shot to increase "
        f"similarity to the reference technique."
    )



if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="LLM Coaching Feedback — test")
    parser.add_argument("--video",    type=str,   help="Run full pipeline then coach")
    parser.add_argument("--hand",     type=str,   default="right")
    parser.add_argument("--mock",     action="store_true",
                        help="Use mock context (no video needed, tests Groq connection)")
    args = parser.parse_args()

    coach = LLMCoach()

    if args.mock:
        # Test with hardcoded context — useful to verify Groq works
        mock_ctx = {
            "shot":             "Cover Drive",
            "shot_confidence":  "84%",
            "technique_overall": 71.0,
            "technique_overall":58.12,
            "pose_flags":["Best Match: straightdrive"],
            "pose_flags":       ["Elbow too closed (88° < 100°) — try opening stance"],
            "length_zone":      "Good length",
            "line_zone":        "Off stump",
            "handedness":       "right",
            "bounce_detected":  True,
        }
        print("\nContext sent to Groq:")
        print(json.dumps(mock_ctx, indent=2))
        tip = coach.generate_from_dict(mock_ctx)

    elif args.video:
        from src.pipeline import Pipeline
        pipe   = Pipeline(handedness=args.hand)
        result = pipe.run(args.video)
        tip    = coach.generate(result)

    else:
        parser.print_help()
        print("\nExample:")
        print("  python src/llm_coach.py --mock")
        print("  python src/llm_coach.py --video clip.mp4")
        exit()

    print(f"\n{'─'*50}")
    print(f"  Coaching Tip  {'(fallback)' if tip.fallback else f'({tip.model_used})'}")
    print(f"{'─'*50}")
    print(tip.tip)
    print()