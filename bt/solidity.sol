// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData {

    struct Student {
        uint id;
        string name;
        uint age;
        uint grade;
    }

    Student[] public students;
    mapping(uint => bool) private studentExists;

    event StudentAdded(uint id, string name, uint age, uint grade);

    function addStudent(uint _id, string memory _name, uint _age, uint _grade) public {
        require(!studentExists[_id], "Student with this ID already exists");
        students.push(Student(_id, _name, _age, _grade));
        studentExists[_id] = true;
        emit StudentAdded(_id, _name, _age, _grade);
    }

    function getStudent(uint _id) public view returns (string memory, uint, uint) {
        for (uint i = 0; i < students.length; i++) {
            if (students[i].id == _id) {
                return (students[i].name, students[i].age, students[i].grade);
            }
        }
        revert("Student not found");
    }

    function getStudentCount() public view returns (uint) {
        return students.length;
    }

    // Receive function to handle plain Ether transfers
    receive() external payable {}

    // Fallback function to handle non-existent function calls
    fallback() external payable {}
}