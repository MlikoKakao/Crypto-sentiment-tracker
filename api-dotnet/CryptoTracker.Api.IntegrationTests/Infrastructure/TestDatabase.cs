using System;
using Microsoft.EntityFrameworkCore;

public class DatabaseFixture : IDisposable
{
    private readonly DbContext _context;