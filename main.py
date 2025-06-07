from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import os, httpx, webbrowser, threading
from pathlib import Path
import sympy as sp
from sympy import symbols, Eq, solve, latex
from sympy.parsing.sympy_parser import parse_expr
import json
##run command  python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
def open_browser():
    def launch():
        webbrowser.open_new("http://localhost:8001/static/whiteboard.html")
    threading.Timer(1.5, launch).start()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

class DrawRequest(BaseModel):
    instruction: str

class MathRequest(BaseModel):
    equation: str

SYSTEM_PROMPT = """
You are an elite AI educational whiteboard agent capable of creating professional, detailed diagrams for any educational content.
Your task: given ANY user request, output ONLY a minified JSON object with a "shapes" array for rendering on an advanced Fabric.js canvas.

AVAILABLE SHAPE TYPES:
- Basic: "circle", "rect", "ellipse", "line", "arrow", "labeledArrow", "text", "mathText", "path", "group"
- Specialized: "cell", "neuron", "molecule", "graph", "chart", "organelle"

PROFESSIONAL STYLING OPTIONS:
- gradient: ["#color1", "#color2", "#color3"] for beautiful gradients
- shadow: {color: "rgba(0,0,0,0.3)", blur: 10, offsetX: 2, offsetY: 2}
- borderRadius: number for rounded corners
- strokeDashArray: [5, 5] for dashed lines
- Professional color palettes available

ENHANCED FEATURES:
- labeledArrow: includes automatic label positioning with leader lines
- mathText: specialized mathematical equation rendering
- cell: automatic organelle placement (specify cellType: "plant" or "animal")
- neuron: complete neuron with dendrites, axon, myelin sheaths, and terminals
- molecule: common molecular structures (specify moleculeType: "water", "glucose", etc.)
- graph: mathematical function plotting with axes and grid
- Professional shadows, gradients, and typography

EXAMPLES:

User: Draw a detailed mitochondria with cristae and matrix labeled
{
  "shapes": [
    {"type":"ellipse","left":200,"top":150,"rx":100,"ry":50,"fill":"#FFE4B5","stroke":"#CD853F","strokeWidth":3,"shadow":{"blur":8,"offsetX":3,"offsetY":3},"originX":"center","originY":"center"},
    {"type":"ellipse","left":200,"top":150,"rx":70,"ry":30,"fill":"#F0E68C","stroke":"#DAA520","strokeWidth":2,"originX":"center","originY":"center"},
    {"type":"path","path":"M 150 140 Q 170 130 190 140 Q 210 150 230 140","stroke":"#8B4513","strokeWidth":2,"fill":""},
    {"type":"path","path":"M 150 160 Q 170 150 190 160 Q 210 170 230 160","stroke":"#8B4513","strokeWidth":2,"fill":""},
    {"type":"path","path":"M 150 180 Q 170 170 190 180 Q 210 190 230 180","stroke":"#8B4513","strokeWidth":2,"fill":""},
    {"type":"labeledArrow","x1":230,"y1":140,"x2":280,"y2":110,"label":"Cristae","stroke":"#333","fontSize":14,"labelBackground":"rgba(255,255,255,0.95)","textShadow":true},
    {"type":"labeledArrow","x1":200,"y1":170,"x2":160,"y2":210,"label":"Matrix","stroke":"#333","fontSize":14,"labelBackground":"rgba(255,255,255,0.95)","textShadow":true},
    {"type":"labeledArrow","x1":250,"y1":150,"x2":320,"y2":120,"label":"Outer Membrane","stroke":"#333","fontSize":12,"labelBackground":"rgba(255,255,255,0.95)"},
    {"type":"text","left":200,"top":220,"text":"Mitochondrion","fontSize":18,"fontWeight":"bold","fill":"#8B4513","originX":"center"}
  ]
}

User: Draw a neuron with dendrites, axon, and synapses labeled
{
  "shapes": [
    {"type":"neuron","left":100,"top":200,"cellType":"motor"},
    {"type":"labeledArrow","x1":80,"y1":180,"x2":50,"y2":150,"label":"Dendrites","stroke":"#2E8B57","fontSize":12},
    {"type":"labeledArrow","x1":125,"y1":200,"x2":160,"y2":230,"label":"Cell Body","stroke":"#2E8B57","fontSize":12},
    {"type":"labeledArrow","x1":200,"y1":200,"x2":230,"y2":170,"label":"Axon","stroke":"#2E8B57","fontSize":12},
    {"type":"labeledArrow","x1":270,"y1":200,"x2":300,"y2":230,"label":"Synapses","stroke":"#2E8B57","fontSize":12}
  ]
}

User: Show the water cycle with evaporation, condensation, and precipitation
{
  "shapes": [
    {"type":"ellipse","left":50,"top":350,"rx":150,"ry":30,"fill":"#4682B4","gradient":["#87CEEB","#4682B4"]},
    {"type":"text","left":120,"top":360,"text":"Ocean","fontSize":16,"fill":"white","fontWeight":"bold"},
    {"type":"path","path":"M 100 320 Q 150 250 200 280 Q 250 250 300 280","stroke":"#87CEEB","strokeWidth":3,"strokeDashArray":[5,5]},
    {"type":"labeledArrow","x1":120,"y1":320,"x2":180,"y2":280,"label":"Evaporation","stroke":"#FF6347","fontSize":12},
    {"type":"ellipse","left":250,"top":100,"rx":80,"ry":40,"fill":"#D3D3D3","gradient":["#F5F5F5","#A9A9A9"]},
    {"type":"text","left":280,"top":110,"text":"Cloud","fontSize":14,"fill":"#333","fontWeight":"bold"},
    {"type":"labeledArrow","x1":300,"y1":140,"x2":330,"y2":180,"label":"Condensation","stroke":"#4169E1","fontSize":12},
    {"type":"path","path":"M 270 140 L 265 180 M 280 140 L 275 180 M 290 140 L 285 180 M 300 140 L 295 180","stroke":"#1E90FF","strokeWidth":3},
    {"type":"labeledArrow","x1":280,"y1":200,"x2":320,"y2":240,"label":"Precipitation","stroke":"#0000CD","fontSize":12}
  ]
}

User: Graph the function y = sin(x) and y = cos(x) on the same axes
{
  "shapes": [
    {"type":"graph","left":100,"top":100,"width":400,"height":300,"title":"Trigonometric Functions"},
    {"type":"path","path":"M 100 250 Q 150 200 200 250 Q 250 300 300 250 Q 350 200 400 250","stroke":"#FF4444","strokeWidth":3,"fill":""},
    {"type":"path","path":"M 100 225 Q 150 175 200 225 Q 250 275 300 225 Q 350 175 400 225","stroke":"#4444FF","strokeWidth":3,"fill":""},
    {"type":"text","left":520,"top":240,"text":"y = sin(x)","fontSize":14,"fill":"#FF4444","fontWeight":"bold"},
    {"type":"text","left":520,"top":215,"text":"y = cos(x)","fontSize":14,"fill":"#4444FF","fontWeight":"bold"},
    {"type":"text","left":300,"top":80,"text":"Trigonometric Functions","fontSize":18,"fontWeight":"bold","fill":"#333"}
  ]
}

User: Draw a plant cell showing nucleus, chloroplasts, and cell wall
{
  "shapes": [
    {"type":"cell","left":150,"top":150,"width":200,"height":150,"cellType":"plant"},
    {"type":"labeledArrow","x1":120,"y1":200,"x2":80,"y2":170,"label":"Cell Wall","stroke":"#228B22","fontSize":12},
    {"type":"labeledArrow","x1":200,"y1":190,"x2":240,"y2":160,"label":"Nucleus","stroke":"#8A2BE2","fontSize":12},
    {"type":"labeledArrow","x1":180,"y1":220,"x2":220,"y2":250,"label":"Chloroplasts","stroke":"#32CD32","fontSize":12}
  ]
}

User: Show photosynthesis equation with reactants and products
{
  "shapes": [
    {"type":"mathText","left":50,"top":150,"text":"6CO₂","fontSize":20,"border":true},
    {"type":"text","left":120,"top":155,"text":"+","fontSize":24,"fontWeight":"bold"},
    {"type":"mathText","left":140,"top":150,"text":"6H₂O","fontSize":20,"border":true},
    {"type":"text","left":200,"top":155,"text":"+","fontSize":24,"fontWeight":"bold"},
    {"type":"mathText","left":220,"top":140,"text":"light","fontSize":16,"fill":"#FF6347"},
    {"type":"text","left":220,"top":160,"text":"energy","fontSize":12,"fill":"#FF6347"},
    {"type":"arrow","x1":280,"y1":155,"x2":350,"y2":155,"stroke":"#333","strokeWidth":3},
    {"type":"mathText","left":370,"top":150,"text":"C₆H₁₂O₆","fontSize":20,"border":true,"fill":"#228B22"},
    {"type":"text","left":470,"top":155,"text":"+","fontSize":24,"fontWeight":"bold"},
    {"type":"mathText","left":490,"top":150,"text":"6O₂","fontSize":20,"border":true,"fill":"#4169E1"},
    {"type":"text","left":200,"top":200,"text":"Photosynthesis","fontSize":18,"fontWeight":"bold","fill":"#2E8B57"},
    {"type":"text","left":180,"top":220,"text":"(in chloroplasts)","fontSize":14,"fill":"#666","fontStyle":"italic"}
  ]
}

INSTRUCTIONS:
- Always use professional colors and styling
- Include appropriate labels and annotations
- Use gradients and shadows for visual appeal
- For biological diagrams, be anatomically accurate
- For mathematical content, use proper notation
- Ensure educational clarity and detail
- No prose, explanations, or code blocks - ONLY the JSON

Create educational diagrams that would impress university professors and engage students!
"""

@app.post("/math_steps")
async def solve_math_steps(req: MathRequest):
    try:
        # Parse the equation
        equation_str = req.equation.strip()
        
        # Handle simple equations like "2x + 5 = 11"
        if '=' in equation_str:
            left_side, right_side = equation_str.split('=', 1)
            left_expr = parse_expr(left_side.strip())
            right_expr = parse_expr(right_side.strip())
            
            # Create equation
            x = symbols('x')
            equation = Eq(left_expr, right_expr)
            
            # Solve step by step
            steps = []
            
            # Step 1: Show original equation
            steps.append({
                "latex": f"${latex(equation)}$",
                "explanation": "Original equation",
                "teaching_comment": ""
            })
            
            # Step 2: Simplify if needed
            simplified = equation.simplify()
            if simplified != equation:
                steps.append({
                    "latex": f"${latex(simplified)}$",
                    "explanation": "Simplified form",
                    "teaching_comment": ""
                })
            
            # Step 3: Solve for x
            solution = solve(equation, x)
            
            # Generate intermediate steps more systematically
            current_equation = equation
            
            # Step-by-step isolation of x
            while len(solve(current_equation, x)) == 1 and current_equation.lhs != x:
                lhs = current_equation.lhs
                rhs = current_equation.rhs
                
                # If left side has addition/subtraction, move constants
                if isinstance(lhs, sp.Add):
                    constants = [term for term in lhs.args if not term.has(x)]
                    if constants:
                        constant = constants[0]
                        if constant > 0:
                            new_lhs = lhs - constant
                            new_rhs = rhs - constant
                            operation = f"Subtract {constant} from both sides"
                        else:
                            new_lhs = lhs - constant  # constant is negative
                            new_rhs = rhs - constant
                            operation = f"Add {abs(constant)} to both sides"
                        
                        current_equation = Eq(new_lhs, new_rhs)
                        steps.append({
                            "latex": f"${latex(current_equation)}$",
                            "explanation": operation,
                            "teaching_comment": ""
                        })
                        continue
                
                # If left side has multiplication, divide both sides
                if isinstance(lhs, sp.Mul):
                    # Find the coefficient of x
                    coeff = None
                    x_term = None
                    for arg in lhs.args:
                        if arg.has(x):
                            x_term = arg
                        elif arg.is_number:
                            coeff = arg
                    
                    if coeff and coeff != 1:
                        new_lhs = lhs / coeff
                        new_rhs = rhs / coeff
                        current_equation = Eq(new_lhs, new_rhs)
                        steps.append({
                            "latex": f"${latex(current_equation)}$",
                            "explanation": f"Divide both sides by {coeff}",
                            "teaching_comment": ""
                        })
                        continue
                
                # If we can't simplify further, break
                break
            
            # Final answer
            if solution:
                steps.append({
                    "latex": f"$x = {latex(solution[0])}$",
                    "explanation": f"Solution: x = {solution[0]}",
                    "teaching_comment": ""
                })
            
            # Generate teaching comments for each step using Groq
            for step in steps:
                if step["explanation"] and step["explanation"] != "Original equation":
                    teaching_prompt = f"Explain this algebra step in simple, encouraging language for a student: {step['explanation']}. Keep it under 20 words and make it friendly."
                    
                    try:
                        payload = {
                            "model": GROQ_MODEL,
                            "messages": [
                                {"role": "system", "content": "You are a friendly math tutor. Explain algebra steps in simple, encouraging language. Keep responses under 20 words."},
                                {"role": "user", "content": teaching_prompt}
                            ],
                            "max_tokens": 100,
                            "temperature": 0.7
                        }
                        headers = {
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        }
                        
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                "https://api.groq.com/openai/v1/chat/completions",
                                json=payload,
                                headers=headers
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                if data.get("choices") and data["choices"][0].get("message"):
                                    step["teaching_comment"] = data["choices"][0]["message"]["content"].strip()
                    except Exception as e:
                        step["teaching_comment"] = "Great job following along with this step!"
            
            return {"steps": steps}
        
        else:
            return {"error": "Please provide an equation with an equals sign (e.g., '2x + 5 = 11')"}
            
    except Exception as e:
        return {"error": f"Error solving equation: {str(e)}"}

@app.post("/groq")
async def proxy_groq(req: DrawRequest):
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.instruction}
        ],
        "max_tokens": 1000,
        "temperature": 0
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            return {"error": "Groq API returned error", "detail": str(e), "body": response.text}

        return response.json()

