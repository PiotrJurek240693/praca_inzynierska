using GBX.NET;
using GBX.NET.Engines.Game;
using GBX.NET.LZO;

namespace MapExtractor
{
    class Point : IEquatable<Point>
    {
        public float X;
        public float Y;

        public Point() { }

        public Point(float x, float y)
        {
            this.X = x;
            this.Y = y;
        }

        public bool Equals(Point? other)
        {
            return this.X == other?.X && this.Y == other?.Y;
        }
    }

    class Line : IEquatable<Line>
    {
        public Point A = new Point();
        public Point B = new Point();

        public bool Equals(Line? other)
        {
            return (this.A.Equals(other?.A) && this.B.Equals(other?.B));
        }

        public Line Reversed()
        {
            return new Line()
            {
                A = this.B,
                B = this.A,
            };
        }

        public void Reverse()
        {
            var temp = this.A;
            this.A = this.B;
            this.B = temp;
        }
    }

    enum BlockType
    {
        Normal,
        Curve,
        Checkpoint,
        Finish,
    }

    class Block
    {
        public Line Enter = new Line();
        public Line Exit = new Line();
        public List<Line> Curve = new List<Line>();
        public BlockType Type;

        public Block() { }
    }

    class Program
    {
        static void Main(string[] args)
        {
            string inputPath = "Test2.Map.Gbx";
            string outputPath = inputPath.Substring(0, inputPath.Length - 3) + "txt";
            Gbx.LZO = new Lzo();
            var map = GameBox.ParseNode<CGameCtnChallenge>(inputPath);

            List<Block> allBlocks = new();

            foreach (var block in map.Blocks)
            {
                if (!block.Name.StartsWith("RoadTech")) continue;

                var xyz = (Vec3)block.Coord * 32;
                string dir = block.Direction.ToString();
                string name = block.Name;

                var exit = new Line();
                var enter = new Line();
                var curve = new List<Line>();

                if (name == "RoadTechStart" || name == "RoadTechFinish" || name == "RoadTechStraight" || name == "RoadTechCheckpoint")
                {
                    if (dir == "North")
                    {
                        enter.A = new Point(xyz.X, xyz.Z);
                        enter.B = new Point(xyz.X + 32, xyz.Z);
                        exit.A = new Point(xyz.X, xyz.Z + 32);
                        exit.B = new Point(xyz.X + 32, xyz.Z + 32);
                    }
                    else if (dir == "South")
                    {
                        enter.A = new Point(xyz.X, xyz.Z);
                        enter.B = new Point(xyz.X + 32, xyz.Z);
                        exit.A = new Point(xyz.X, xyz.Z + 32);
                        exit.B = new Point(xyz.X + 32, xyz.Z + 32);
                    }
                    else if (dir == "East")
                    {
                        enter.A = new Point(xyz.X + 32, xyz.Z);
                        enter.B = new Point(xyz.X + 32, xyz.Z + 32);
                        exit.A = new Point(xyz.X, xyz.Z);
                        exit.B = new Point(xyz.X, xyz.Z + 32);
                    }
                    else // dir == "West"
                    {
                        enter.A = new Point(xyz.X + 32, xyz.Z);
                        enter.B = new Point(xyz.X + 32, xyz.Z + 32);
                        exit.A = new Point(xyz.X, xyz.Z);
                        exit.B = new Point(xyz.X, xyz.Z + 32);
                    }
                }
                else if (name.StartsWith("RoadTechCurve"))
                {
                    int r = 0;
                    if (name.Contains("2")) r = 1;
                    else if (name.Contains("3")) r = 2;

                    if (dir == "North")
                    {
                        enter.A = new Point(xyz.X, xyz.Z);
                        enter.B = new Point(xyz.X, xyz.Z + 32);
                        exit.A = new Point(xyz.X + 32 + r * 32, xyz.Z + 32 + r * 32);
                        exit.B = new Point(xyz.X + r * 32, xyz.Z + 32 + r * 32);
                    }
                    else if (dir == "South")
                    {
                        exit.A = new Point(xyz.X, xyz.Z);
                        exit.B = new Point(xyz.X + 32, xyz.Z);
                        enter.A = new Point(xyz.X + 32 + r * 32, xyz.Z + 32 + r * 32);
                        enter.B = new Point(xyz.X + 32 + r * 32, xyz.Z + r * 32);
                    }
                    else if (dir == "East")
                    {
                        enter.B = new Point(xyz.X + r * 32, xyz.Z);
                        enter.A = new Point(xyz.X + 32 + r * 32, xyz.Z);
                        exit.B = new Point(xyz.X, xyz.Z + r * 32);
                        exit.A = new Point(xyz.X, xyz.Z + 32 + r * 32);
                    }
                    else // dir == "West"
                    {
                        enter.A = new Point(xyz.X, xyz.Z + 32 + r * 32);
                        enter.B = new Point(xyz.X + 32, xyz.Z + 32 + r * 32);
                        exit.A = new Point(xyz.X + 32 + r * 32, xyz.Z);
                        exit.B = new Point(xyz.X + 32 + r * 32, xyz.Z + 32);
                    }

                    Point center;
                    double q;
                    if (dir == "North")
                    {
                        center = new Point(xyz.X, xyz.Z + 32 * (r + 1));
                        q = 1.5;
                    }
                    else if (dir == "East")
                    {
                        center = new Point(xyz.X, xyz.Z);
                        q = 0;
                    }
                    else if (dir == "South")
                    {
                        center = new Point(xyz.X + 32 * (r + 1), xyz.Z);
                        q = 0.5;
                    }
                    else
                    {
                        center = new Point(xyz.X + 32 * (r + 1), xyz.Z + 32 * (r + 1));
                        q = 1;
                    }
                    double theta = 0.0;
                    while (theta <= 0.5)
                    {
                        double x = center.X + (r + 1) * 32 * Math.Cos((theta + q) * Math.PI);
                        double y = center.Y + (r + 1) * 32 * Math.Sin((theta + q) * Math.PI);
                        double x1 = center.X + r * 32 * Math.Cos((theta + q) * Math.PI);
                        double y1 = center.Y + r * 32 * Math.Sin((theta + q) * Math.PI);
                        curve.Add(new Line()
                        {
                            A = new Point()
                            {
                                X = (float)x,
                                Y = (float)y,
                            },
                            B = new Point()
                            {
                                X = (float)x1,
                                Y = (float)y1,
                            }
                        });
                        theta += 0.05;
                    }
                }

                BlockType type = BlockType.Normal;
                if (name.StartsWith("RoadTechCurve"))
                {
                    type = BlockType.Curve;
                }
                else if (name.Contains("Checkpoint"))
                {
                    type = BlockType.Checkpoint;
                }
                else if (name.Contains("Finish"))
                {
                    type = BlockType.Finish;
                }

                var newBlock = new Block()
                {
                    Enter = enter,
                    Exit = exit,
                    Type = type,
                    Curve = curve,
                };

                if (name == "RoadTechStart")
                    allBlocks.Insert(0, newBlock);
                else
                    allBlocks.Add(newBlock);
            }

            for (int i = 0; i < allBlocks.Count - 1; i++)
            {
                var current = allBlocks[i];

                var ok = false;
                for (int j = i + 1; j < allBlocks.Count; j++)
                {
                    var next = allBlocks[j];

                    if (current.Exit.Equals(next.Enter))
                    {
                        var tmp = allBlocks[i + 1];
                        allBlocks[i + 1] = allBlocks[j];
                        allBlocks[j] = tmp;
                        ok = true;
                        break;
                    }
                    else if (current.Exit.Equals(next.Exit))
                    {
                        var tmpLine = next.Exit;
                        next.Exit = next.Enter;
                        next.Enter = tmpLine;
                        if (next.Curve.Count > 0)
                        {
                            next.Curve.Reverse();
                        }
                        var tmp = allBlocks[i + 1];
                        allBlocks[i + 1] = allBlocks[j];
                        allBlocks[j] = tmp;
                        ok = true;
                        break;
                    }
                    else if (current.Exit.Equals(next.Enter.Reversed()))
                    {
                        next.Enter.Reverse();
                        next.Exit.Reverse();
                        foreach (var x in next.Curve)
                        {
                            x.Reverse();
                        }
                        var tmp = allBlocks[i + 1];
                        allBlocks[i + 1] = allBlocks[j];
                        allBlocks[j] = tmp;
                        ok = true;
                        break;
                    }
                    else if (current.Exit.Equals(next.Exit.Reversed()))
                    {
                        next.Enter.Reverse();
                        next.Exit.Reverse();
                        foreach (var x in next.Curve)
                        {
                            x.Reverse();
                        }
                        var tmpLine = next.Exit;
                        next.Exit = next.Enter;
                        next.Enter = tmpLine;
                        if (next.Curve.Count > 0)
                        {
                            next.Curve.Reverse();
                        }
                        var tmp = allBlocks[i + 1];
                        allBlocks[i + 1] = allBlocks[j];
                        allBlocks[j] = tmp;
                        ok = true;
                        break;
                    }
                }

                if (!ok)
                {
                    throw new Exception("Nie znaleziono kolejnej czesci");
                }
            }

            using StreamWriter file = new(outputPath, false);
            for (int i = 0; i < allBlocks.Count; i++)
            {
                var block = allBlocks[i];

                var blockEnter = block.Enter;
                var blockExit = block.Exit;

                if (block.Type == BlockType.Checkpoint || block.Type == BlockType.Finish)
                {
                    if (blockEnter.A.X == blockEnter.B.X)
                    {
                        var change = blockExit.A.X > blockEnter.B.X ? 6 : -6;
                        blockEnter.A.X += change;
                        blockEnter.B.X += change;
                    }
                    else // blockEnter.a.y == blockEnter.b.y
                    {
                        var change = blockExit.A.Y > blockEnter.B.Y ? 6 : -6;
                        blockEnter.A.Y += change;
                        blockEnter.B.Y += change;

                    }
                }

                //if (blockEnter.a.x == blockEnter.b.x)
                //{
                //    if (blockEnter.a.y > blockEnter.b.y)
                //    {
                //        blockEnter.a.y -= 6;
                //        blockEnter.b.y += 6;
                //    }
                //    else
                //    {
                //        blockEnter.a.y += 6;
                //        blockEnter.b.y -= 6;
                //    }
                //}
                //else
                //{
                //    if (blockEnter.a.x > blockEnter.b.x)
                //    {
                //        blockEnter.a.x -= 6;
                //        blockEnter.b.x += 6;
                //    }
                //    else
                //    {
                //        blockEnter.a.x += 6;
                //        blockEnter.b.x -= 6;
                //    }
                //}

                file.WriteLine($"{blockEnter.A.X} {blockEnter.A.Y} {blockEnter.B.X} {blockEnter.B.Y} {(int)block.Type}");

                if (block.Type == BlockType.Curve)
                {
                    foreach (var points in block.Curve)
                    {
                        file.WriteLine($"{points.A.X} {points.A.Y} {points.B.X} {points.B.Y} {(int)block.Type}");
                    }
                }
            }

            Console.WriteLine($"{outputPath}");
        }
    }
}
