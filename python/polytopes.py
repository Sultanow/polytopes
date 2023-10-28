from cytools import Polytope
    
if __name__ == "__main__":
    vertices = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]]
    p = Polytope(vertices)
    print(p.faces(d=2)[0].points())