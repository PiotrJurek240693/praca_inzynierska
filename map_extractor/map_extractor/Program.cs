using System;
using GBX.NET;
using GBX.NET.Engines.Game;
using System.Collections.Generic;
using System.IO;
using GBX.NET.LZO;

namespace MapExtractor
{
    class Point : IEquatable<Point>
    {
        public float x;
        public float y;

        public Point() { }

        public Point(float x, float y)
        {
            this.x = x;
            this.y = y;
        }

        public bool Equals(Point? other)
        {
            return this.x == other?.x && this.y == other?.y;
        }
    }

    class Line : IEquatable<Line>
    {
        public Point a = new Point();
        public Point b = new Point();

        public bool Equals(Line? other)
        {
            return (this.a.Equals(other?.a) && this.b.Equals(other?.b))
                || (this.a.Equals(other?.b) && this.b.Equals(other?.a));
        }
    }

    enum BlockType
    {
        Normal,
        Checkpoint,
        Finish,
    }

    class Block
    {
        public Line enter = new Line();
        public Line exit = new Line();
        public BlockType type;

        public Block() { }

        public Block(Line enter, Line exit, BlockType type)
        {
            this.enter = enter;
            this.exit = exit;
            this.type = type;
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            string inputPath = "Test.Map.Gbx";
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

                var arr_l = new Line();
                var arr_r = new Line();

                if (name == "RoadTechStart" || name == "RoadTechFinish" || name == "RoadTechStraight" || name == "RoadTechCheckpoint")
                {
                    if (dir == "North")
                    {
                        arr_r.a = new Point(xyz.X, xyz.Z);
                        arr_r.b = new Point(xyz.X + 32, xyz.Z);
                        arr_l.a = new Point(xyz.X, xyz.Z + 32);
                        arr_l.b = new Point(xyz.X + 32, xyz.Z + 32);
                    }
                    else if (dir == "South")
                    {
                        arr_r.a = new Point(xyz.X, xyz.Z);
                        arr_r.b = new Point(xyz.X + 32, xyz.Z);
                        arr_l.a = new Point(xyz.X, xyz.Z + 32);
                        arr_l.b = new Point(xyz.X + 32, xyz.Z + 32);
                    }
                    else if (dir == "East")
                    {
                        arr_r.a = new Point(xyz.X + 32, xyz.Z);
                        arr_r.b = new Point(xyz.X + 32, xyz.Z + 32);
                        arr_l.a = new Point(xyz.X, xyz.Z);
                        arr_l.b = new Point(xyz.X, xyz.Z + 32);
                    }
                    else // dir == "West"
                    {
                        arr_r.a = new Point(xyz.X + 32, xyz.Z);
                        arr_r.b = new Point(xyz.X + 32, xyz.Z + 32);
                        arr_l.a = new Point(xyz.X, xyz.Z);
                        arr_l.b = new Point(xyz.X, xyz.Z + 32);
                    }
                }
                else if (name.StartsWith("RoadTechCurve"))
                {
                    int r = 0;
                    if (name.Contains("2")) r = 1;
                    else if (name.Contains("3")) r = 2;

                    if (dir == "North")
                    {
                        arr_r.a = new Point(xyz.X, xyz.Z);
                        arr_r.b = new Point(xyz.X, xyz.Z + 32);
                        arr_l.a = new Point(xyz.X + r * 32, xyz.Z + 32 + r * 32);
                        arr_l.b = new Point(xyz.X + 32 + r * 32, xyz.Z + 32 + r * 32);
                    }
                    else if (dir == "East")
                    {
                        arr_r.a = new Point(xyz.X + r * 32, xyz.Z);
                        arr_r.b = new Point(xyz.X + 32 + r * 32, xyz.Z);
                        arr_l.a = new Point(xyz.X, xyz.Z + r * 32);
                        arr_l.b = new Point(xyz.X, xyz.Z + 32 + r * 32);
                    }
                    else if (dir == "South")
                    {
                        arr_r.a = new Point(xyz.X, xyz.Z);
                        arr_r.b = new Point(xyz.X + 32, xyz.Z);
                        arr_l.a = new Point(xyz.X + 32 + r * 32, xyz.Z + r * 32);
                        arr_l.b = new Point(xyz.X + 32 + r * 32, xyz.Z + 32 + r * 32);
                    }
                    else // dir == "West"
                    {
                        arr_r.a = new Point(xyz.X, xyz.Z + 32 + r * 32);
                        arr_r.b = new Point(xyz.X + 32, xyz.Z + 32 + r * 32);
                        arr_l.a = new Point(xyz.X + 32 + r * 32, xyz.Z);
                        arr_l.b = new Point(xyz.X + 32 + r * 32, xyz.Z + 32);
                    }
                }

                BlockType type = BlockType.Normal;
                if (name.Contains("Checkpoint"))
                {
                    type = BlockType.Checkpoint;
                }
                else if (name.Contains("Finish"))
                {
                    type = BlockType.Finish;
                }

                var newBlock = new Block(arr_r, arr_l, type);

                if (name == "RoadTechStart")
                    allBlocks.Insert(0, newBlock);
                else
                    allBlocks.Add(newBlock);
            }

            for (int i = 0; i < allBlocks.Count - 1; i++)
            {
                var current = allBlocks[i];

                for (int j = i + 1; j < allBlocks.Count; j++)
                {
                    var next = allBlocks[j];

                    if (current.exit.Equals(next.enter))
                    {
                        var tmp = allBlocks[i + 1];
                        allBlocks[i + 1] = allBlocks[j];
                        allBlocks[j] = tmp;
                        break;
                    }
                    else if (current.exit.Equals(next.exit))
                    {
                        var tmpLine = next.exit;
                        next.exit = next.enter;
                        next.enter = tmpLine;
                        var tmp = allBlocks[i + 1];
                        allBlocks[i + 1] = allBlocks[j];
                        allBlocks[j] = tmp;
                        break;
                    }
                }
            }

            using StreamWriter file = new(outputPath);
            for (int i = 1; i < allBlocks.Count; i++)
            {
                var block = allBlocks[i];
                var blockEnter = block.enter;
                var blockExit = block.enter;

                if (block.type == BlockType.Checkpoint || block.type == BlockType.Finish)
                {
                    if (blockEnter.a.x == blockEnter.b.x)
                    {
                        var change = blockExit.a.x > blockEnter.b.x ? 6 : -6;
                        blockEnter.a.x += change;
                        blockEnter.b.x += change;
                    }
                    else // blockEnter.a.y == blockEnter.b.y
                    {
                        var change = blockExit.a.y > blockEnter.b.y ? 6 : -6;
                        blockEnter.a.y += change;
                        blockEnter.b.y += change;

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

                file.WriteLine($"{blockEnter.a.x} {blockEnter.a.y} {blockEnter.b.x} {blockEnter.b.y} {(int)block.type}");
            }

            Console.WriteLine($"{outputPath}");
        }
    }
}
